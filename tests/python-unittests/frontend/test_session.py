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
"""Conversation history, regeneration, cancellation, and recovery."""

import io
import signal
import types

import pytest

from experimental import server
from experimental.frontend import backend, cli, session, terminal
from experimental.server.parsing import reasoning
from experimental.server.runtime import engine

from . import helpers


@pytest.mark.parametrize("interrupted,text", [(False, "replacement"),
                                              (True, "partial"), (True, "")])
def test_regenerate_replaces_last_response_transactionally(interrupted, text):
    # Given an existing answer.
    # When regeneration completes or stops, then only a usable replacement changes
    # history.
    model = helpers.FakeBackend()
    conversation = session.Session(model, [{
        "role": "system",
        "content": "Be brief."
    }])
    conversation.generate("first", lambda _: None)
    conversation.generate("second", lambda _: None)
    original = conversation.history[:]
    model.text, model.interrupt = text, interrupted
    conversation.regenerate(lambda _: None)
    assert model.last_messages == original[:-1]
    expected = original[:-1] + [{
        "role": "assistant",
        "content": text
    }] if text else original
    assert conversation.history == expected
    assert conversation.state == "ready" and model.drained
    assert conversation.interrupted == interrupted
    model.error = ValueError("rejected")
    with pytest.raises(ValueError, match="rejected"):
        conversation.regenerate(lambda _: None)
    assert conversation.history == expected


def test_regenerate_requires_a_response_and_clear_resets_it():
    # Given no retained answer.
    # When regeneration is requested, then it fails even after clearing a
    # conversation.
    conversation = session.Session(helpers.FakeBackend(), [])
    with pytest.raises(ValueError, match="No response"):
        conversation.regenerate(lambda _: None)
    conversation.generate("hello", lambda _: None)
    conversation.command("/clear")
    with pytest.raises(ValueError, match="No response"):
        conversation.regenerate(lambda _: None)


@pytest.mark.parametrize("regenerate", [False, True])
def test_native_stream_error_preserves_history_and_allows_recovery(
        tmp_path, regenerate):
    # Given a stream failure.
    # When generating or regenerating, then cleanup preserves history and permits
    # a later turn.
    model = backend.Backend.__new__(backend.Backend)
    model.sampling = engine.SamplingParams()
    drained = []
    fail = True

    def generate(*_args, **_kwargs):

        def chunks():
            try:
                yield engine.StreamDelta(text="partial" if fail else "answer")
                yield engine.StreamDelta(
                    finished=True, finish_reason="error" if fail else "stop")
            finally:
                drained.append(True)

        return server.CancellableIterator(chunks(), lambda: None)

    model.llm = types.SimpleNamespace(model_dir=str(tmp_path),
                                      generate_stream=generate,
                                      validate_context=lambda *_: 3,
                                      max_kv_cache_capacity=8192,
                                      close=lambda: None)
    conversation = session.Session(model, [])
    history = [{
        "role": "user",
        "content": "old question"
    }, {
        "role": "assistant",
        "content": "old answer"
    }]
    conversation.history = history[:]
    with pytest.raises(RuntimeError, match="Generation failed in the backend"):
        if regenerate:
            conversation.regenerate(lambda _: None)
        else:
            conversation.generate("new question", lambda _: None)
    assert drained == [True]
    assert conversation.history == history
    assert conversation.state == "ready"
    fail = False
    conversation.generate("try again", lambda _: None)
    assert conversation.history[-1]["content"] == "answer"
    conversation.close()
    assert conversation.state == "closed"


@pytest.mark.parametrize("cancel", [False, True])
def test_thinking_display_history_and_regeneration(tmp_path, cancel):
    # Given reasoning output.
    # When a turn finishes or stops, then rendering and subsequent requests retain
    # it separately.
    model = helpers.FakeBackend()
    requests = []

    def stream(messages, on_metrics):
        requests.append(messages)

        def iterate():
            yield reasoning.ReasoningDelta("reasoning", "Let me check.")
            if cancel:
                signal.raise_signal(signal.SIGINT)
            else:
                yield reasoning.ReasoningDelta("content", "Four.")

        return server.CancellableIterator(iterate(), lambda: None)

    model.stream = stream
    conversation = session.Session(model, [])
    output = io.StringIO()
    display = terminal.Terminal(output, helpers.settings(tmp_path))
    cli._turn(conversation, display, "What is 2+2?")
    rendered = output.getvalue()
    assert (rendered.index("[Start thinking]") <
            rendered.index("Let me check.") < rendered.index("[End thinking]"))
    if cancel:
        assert "Four." not in rendered
    else:
        assert rendered.index("[End thinking]") < rendered.index("Four.")
    assert "\033" not in rendered
    answer = {
        "role": "assistant",
        "content": "" if cancel else "Four.",
        "reasoning_content": "Let me check."
    }
    assert conversation.history[-1] == answer
    cli._turn(conversation, display, "/regen", regenerate=True)
    assert requests[1] == requests[0]
    conversation.generate("Next", lambda _: None, lambda _: None)
    assert requests[-1][-2] == answer


@pytest.mark.parametrize("text", ["partial", ""])
def test_cancel_keeps_visible_history_and_final_metrics(text):
    # Given an interrupted turn.
    # When cleanup finishes, then visible history and final metrics survive for
    # the next turn.
    model = helpers.FakeBackend()
    model.interrupt, model.text = True, text
    initial = [{"role": "system", "content": "Be brief."}]
    conversation = session.Session(model, initial)
    output = io.StringIO()
    original_handler = signal.getsignal(signal.SIGINT)
    conversation.generate("hello", output.write)
    assert output.getvalue() == text
    assert model.cancelled and model.drained
    assert conversation.metrics.prompt_tokens == 3
    assert conversation.metrics.generated_tokens == 1
    assert conversation.metrics.time_to_first_token == 0.1
    assert conversation.metrics.total_time == 0.2
    assert conversation.state == "ready"
    assert conversation.interrupted
    assert signal.getsignal(signal.SIGINT) == original_handler
    assert conversation.history == initial + ([
        {
            "role": "user",
            "content": "hello"
        },
        {
            "role": "assistant",
            "content": text
        },
    ] if text else [])
    model.interrupt, model.text = False, "next"
    conversation.generate("continue", output.write)
    assert conversation.history[-1]["content"] == "next"
    conversation.command("/clear")
    assert conversation.history == initial
    conversation.close()
    assert model.closed and conversation.state == "closed"


def test_rejected_turn_preserves_history():
    # Given a context-limit rejection.
    # When a new turn is attempted, then history remains unchanged and the session
    # stays ready.
    model = helpers.FakeBackend()
    conversation = session.Session(model, [])
    conversation.generate("hello", lambda _: None)
    history = conversation.history[:]
    model.error = ValueError("context limit")
    with pytest.raises(ValueError, match="context limit"):
        conversation.generate("too long", lambda _: None)
    assert conversation.history == history
    assert conversation.state == "ready"
