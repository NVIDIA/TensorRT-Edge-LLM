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
"""Completion, display labels, color, and loading progress."""

import io
import os
import re
import threading

import pytest
from prompt_toolkit import completion, document

from experimental.frontend import session, terminal
from experimental.server.runtime import engine

from . import helpers


def test_runtime_details_labels_and_values(tmp_path, capsys):
    # Given runtime details.
    # When displayed, then plain labels and values go only to stderr.
    responses = io.StringIO()
    details = {
        "model": "Qwen",
        "device": "NVIDIA Thor",
        "weights": "BF16 (checkpoint)",
        "speculative mode": "off"
    }
    terminal.Terminal(responses,
                      helpers.settings(tmp_path)).runtime_details(details)
    stdout, stderr = capsys.readouterr()
    assert not stdout and not responses.getvalue()
    assert "\033" not in stderr
    for label, value in details.items():
        assert re.search(rf"\b{label}\s*:\s*{re.escape(value)}", stderr)


def test_prompt_completion_commands_and_plain_text():
    # Given command prefixes or prose.
    # When completing, then only supported leading commands are suggested.
    completer = terminal.PromptCompleter()
    event = completion.CompleteEvent(completion_requested=True)

    def matches(text):
        return list(completer.get_completions(document.Document(text), event))

    assert {item.text
            for item in matches("/")
            } == {"/help", "/clear", "/regen", "/file ", "/exit"}
    match, = matches("/cl")
    assert (match.text, match.start_position) == ("/clear", -3)
    for text in ("", "hello", "explain /cl", "/unknown", "hello\n/cl"):
        assert matches(text) == []


def test_prompt_completion_file_paths(tmp_path, monkeypatch):
    # Given file names with spaces.
    # When completing a file command, then both relative and home paths work.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "notes with spaces.txt").write_text("hello")
    completer = terminal.PromptCompleter()
    event = completion.CompleteEvent(completion_requested=True)
    for prefix in ("/file notes with", "/file ~/notes with"):
        match, = completer.get_completions(document.Document(prefix), event)
        assert prefix + match.text == prefix + " spaces.txt"
        assert match.start_position == 0


def test_banner_colors_respect_no_color(tmp_path, monkeypatch, capsys):
    # Given an interactive terminal.
    # When NO_COLOR is set, then the banner stays readable without escape codes.
    output = io.StringIO()
    monkeypatch.setattr(output, "isatty", lambda: True)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    terminal.Terminal(output, helpers.settings(tmp_path)).startup()
    colored = capsys.readouterr()
    assert not colored.out and not output.getvalue()
    assert terminal.GREEN in colored.err
    monkeypatch.setenv("NO_COLOR", "1")
    terminal.Terminal(output, helpers.settings(tmp_path)).startup()
    plain = capsys.readouterr()
    assert not plain.out and not output.getvalue()
    assert "\033" not in plain.err
    for label in ("Edge-LLM", "/help", "Ctrl+C"):
        assert label in plain.err


@pytest.mark.parametrize("fail", [False, True])
def test_loading_refresh_stops_after_success_or_error(tmp_path, monkeypatch,
                                                      fail):
    # Given active loading.
    # When it succeeds or fails, then progress refresh stops and its thread exits.
    display = terminal.Terminal(io.StringIO(), helpers.settings(tmp_path))
    refreshed = threading.Event()
    threads = []

    def status(text):
        if threading.current_thread() is not threading.main_thread():
            threads.append(threading.current_thread())
            assert "Building text engine (1/2)" in text
            refreshed.set()

    monkeypatch.setattr(display, "status", status)
    monkeypatch.setattr(terminal, "PROGRESS_INTERVAL_SECONDS", 0.01)
    try:
        with display.loading():
            display.progress("Building text engine (1/2)")
            assert refreshed.wait(2)
            if fail:
                raise ValueError("build failed")
    except ValueError:
        assert fail
    assert threads and all(not thread.is_alive() for thread in threads)


def test_loading_stages_keep_native_logs_on_separate_lines(tmp_path, capfd):
    # Given native diagnostics between progress stages.
    # When rendered, then each message occupies its own line.
    display = terminal.Terminal(io.StringIO(), helpers.settings(tmp_path))
    display.color = True
    with display.loading():
        display.progress("Building engine")
        os.write(2, b"Native build diagnostic\n")
        display.progress("Ready")
    display.status("Next message")
    stdout, stderr = capfd.readouterr()
    assert stdout == ""
    lines = [line for line in stderr.splitlines() if line.strip()]
    assert len(lines) == 4
    for line, label in zip(lines,
                           ("Building engine", "Native build diagnostic",
                            "Ready", "Next message")):
        assert label in line
    assert stderr.endswith("\n")
    assert "\r" not in stderr and "\033[K" not in stderr


@pytest.mark.parametrize("prompt_tokens, expected", [
    (157, "331/8.19K (4%)"),
    (None, "n/a/8.19K"),
])
def test_metrics_show_full_context_and_timings_by_default(
        tmp_path, capsys, prompt_tokens, expected):
    # Given final native metrics.
    # When a response ends, then stderr reports rates, context use, and latency.
    model = helpers.FakeBackend()
    conversation = session.Session(model, [])
    conversation.metrics = engine.GenerationMetrics(
        prompt_tokens=prompt_tokens,
        processed_prompt_tokens=10,
        prompt_time=0.1,
        generated_tokens=174,
        decoded_tokens=173,
        generation_time=36.0,
        time_to_first_token=0.262,
        total_time=36.915)
    display = terminal.Terminal(io.StringIO(), helpers.settings(tmp_path))
    display.finish_response(conversation)
    stdout, stderr = capsys.readouterr()
    assert stdout == ""
    assert "\033" not in stderr
    summary = " ".join(stderr.split())
    for field in ("prompt 100.0 tok/s", "generation 4.8 tok/s",
                  f"context {expected}", "TTFT 0.262s", "total 36.915s"):
        assert field in summary
