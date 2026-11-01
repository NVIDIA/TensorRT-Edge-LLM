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
"""CLI entry points, shell behavior, and one interactive smoke test."""

import io
import json
import os
import subprocess
import sys
import textwrap
import threading
import types

import prompt_toolkit
import pytest
from prompt_toolkit import application, input, output

from experimental.frontend import backend, cli, config, terminal
from tensorrt_edgellm import cli as commands
from tensorrt_edgellm import cli_version as version

from . import helpers


@pytest.mark.parametrize("checkout", [False, True])
def test_version_uses_metadata_without_loading_model(tmp_path, monkeypatch,
                                                     capsys, checkout):
    # Given package metadata.
    # When version is requested, then it reports the version and optional revision
    # without loading a model.
    monkeypatch.setattr(version, "SOURCE_ROOT", tmp_path)
    monkeypatch.setattr(version.metadata, "version", lambda _: "1.2.3")
    if checkout:
        (tmp_path / ".git").write_text("gitdir: /worktree")

    def git(*_args, **_kwargs):
        assert checkout
        return types.SimpleNamespace(stdout="123456789\n")

    monkeypatch.setattr(version.subprocess, "run", git)
    monkeypatch.setattr(backend, "Backend",
                        lambda *_: pytest.fail("loaded model"))
    with pytest.raises(SystemExit) as result:
        commands.main(["--version"])
    assert result.value.code == 0
    stdout, stderr = capsys.readouterr()
    assert "edge-llm" in stdout and "1.2.3" in stdout
    assert ("123456789" in stdout) == checkout
    assert not stderr


def test_version_falls_back_without_metadata_or_git(tmp_path, monkeypatch):
    # Given missing metadata and git.
    # When version is requested, then the bundled version remains available.
    from tensorrt_edgellm._version import __version__

    monkeypatch.setattr(version, "SOURCE_ROOT", tmp_path)
    (tmp_path / ".git").mkdir()

    def missing_metadata(_name):
        raise version.metadata.PackageNotFoundError

    def missing_git(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(version.metadata, "version", missing_metadata)
    monkeypatch.setattr(version.subprocess, "run", missing_git)
    text = version.version_string()
    assert "edge-llm" in text and __version__ in text
    assert "git" not in text


def test_chat_help_lists_option_groups(capsys):
    # Given a help request.
    # When parsing, then all launch-option groups are shown with a successful
    # exit.
    with pytest.raises(SystemExit) as result:
        config.parser().parse_args(["--help"])
    assert result.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "trtedge chat" in help_text
    for title, flag in (("Input and configuration",
                         "--prompt"), ("Generation", "--temperature"),
                        ("Thinking and speculation", "--speculative-config"),
                        ("Engine and cache", "--max-input-len"),
                        ("Output and diagnostics", "--verbose")):
        assert title in help_text and flag in help_text


@pytest.mark.parametrize("specification", [
    {
        "method": []
    },
    {
        "method": "mtp",
        "num_speculative_tokens": "3"
    },
])
def test_malformed_speculation_reports_configuration_error(
        tmp_path, monkeypatch, capsys, specification):
    # Given malformed speculation settings.
    # When the CLI starts, then it reports a usage error without loading a model.
    monkeypatch.setattr(backend, "Backend",
                        lambda *_: pytest.fail("loaded model"))
    with pytest.raises(SystemExit) as result:
        commands.main([
            "chat",
            str(tmp_path), "--speculative-config",
            json.dumps(specification)
        ])
    assert result.value.code == 2
    stdout, stderr = capsys.readouterr()
    assert not stdout
    assert "error:" in stderr and "--speculative-config." in stderr
    assert "Traceback" not in stderr


def test_one_shot_keeps_native_and_python_logs_off_stdout(
        tmp_path, monkeypatch, capfd):
    # Given a one-shot response and diagnostics.
    # When the CLI runs, then stdout contains only the answer.
    model = helpers.FakeBackend()

    def load(*_args):
        print("Python diagnostic")
        os.write(1, b"Native diagnostic\n")
        return model

    monkeypatch.setattr(backend, "Backend", load)
    assert commands.main(["chat", str(tmp_path), "--prompt", "hello"]) == 0
    stdout, stderr = capfd.readouterr()
    assert stdout == "answer\n"
    assert "Python diagnostic" in stderr and "Native diagnostic" in stderr
    assert "tok/s" in stderr
    assert "\033" not in stdout + stderr
    assert model.closed


@pytest.mark.parametrize("fail", [False, True])
def test_buffered_native_logs_are_flushed_before_stdout_restoration(fail):
    # Given buffered native output.
    # When diagnostics close normally or on error, then logs stay off response
    # stdout.
    script = textwrap.dedent("""\
        import ctypes
        import sys
        from experimental.frontend.logging import diagnostics

        libc = ctypes.CDLL(None)
        libc.setvbuf.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_int, ctypes.c_size_t]
        libc.printf.argtypes = [ctypes.c_char_p]
        libc.fflush.argtypes = [ctypes.c_void_p]
        stdout = ctypes.c_void_p.in_dll(libc, "stdout")
        assert libc.setvbuf(stdout, None, 0, 4096) == 0
        try:
            with diagnostics() as responses:
                libc.printf(b"Buffered native diagnostic")
                print("answer", file=responses)
                if sys.argv[1] == "True":
                    raise RuntimeError("test cleanup")
        except RuntimeError:
            pass
        libc.fflush(None)
        print("restored")
        """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(fail)],
        capture_output=True,
        text=True,
        timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "answer\nrestored\n"
    assert result.stderr == "Buffered native diagnostic"


def test_interactive_commands_file_input_and_eof(tmp_path, monkeypatch,
                                                 capsys):
    # Given file and lifecycle commands.
    # When an interactive session reaches EOF, then responses, errors, and cleanup
    # are correct.
    path = tmp_path / "prompt.txt"
    path.write_text("from file")
    model = helpers.FakeBackend()
    monkeypatch.setattr(backend, "Backend", lambda *_: model)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(f"\n/unknown\n/file {tmp_path / 'missing.txt'}\n"
                    f"/help\n/regen\n/file {path}\n/regen\n/clear\n"))
    output = io.StringIO()
    assert cli.run(helpers.settings(tmp_path), output) == 0
    assert output.getvalue() == "answer\nanswer\n"
    messages = capsys.readouterr().err
    assert "Conversation cleared" in messages
    assert "No response to regenerate" in messages
    assert "Unknown command" in messages and "Error:" in messages
    assert model.closed


@pytest.mark.parametrize("error,verbose,exit_code,label", [
    (KeyboardInterrupt(), False, 130, "Stopped"),
    (BrokenPipeError(), False, 0, None),
    (RuntimeError("load failed"), False, 1, "Error:"),
    (RuntimeError("load failed"), True, 1, "Traceback"),
])
def test_shell_exit_status_and_error_routing(tmp_path, monkeypatch, capfd,
                                             error, verbose, exit_code, label):

    # Given an interrupted or failed startup.
    # When the CLI exits, then shell status and stderr match the failure mode.
    def load(*_args):
        raise error

    monkeypatch.setattr(backend, "Backend", load)
    options = ["--verbose"] if verbose else []
    assert commands.main(
        ["chat", str(tmp_path), "--prompt", "hello", *options]) == exit_code
    stdout, stderr = capfd.readouterr()
    assert not stdout
    if label:
        assert label in stderr
    else:
        assert not stderr
    if not verbose:
        assert "Traceback" not in stderr


def test_one_shot_cancellation_returns_shell_status(tmp_path, monkeypatch,
                                                    capfd):
    # Given one-shot cancellation.
    # When the CLI exits, then partial output survives and status is 130 after
    # cleanup.
    model = helpers.FakeBackend()
    model.text, model.interrupt = "partial", True
    monkeypatch.setattr(backend, "Backend", lambda *_: model)
    assert commands.main(["chat", str(tmp_path), "--prompt", "hello"]) == 130
    stdout, stderr = capfd.readouterr()
    assert stdout == "partial\n"
    assert "stopped" in stderr.lower() and "total" in stderr
    assert model.cancelled and model.drained and model.closed


def test_terminal_completion_cancel_and_next_turn(tmp_path, monkeypatch):
    # Given an interactive session.
    # When completion and cancellation occur, then the next turn works with
    # retained partial history.
    responses = io.StringIO()
    display = terminal.Terminal(responses, helpers.settings(tmp_path))
    display.interactive = True
    model = helpers.FakeBackend()
    model.text, model.interrupt = "partial", True
    model.details = lambda: {}
    statuses = []
    prompts = []
    with input.create_pipe_input() as pipe:
        prompt_session = prompt_toolkit.PromptSession

        def make_prompt(*args, **kwargs):
            prompt = prompt_session(*args, **kwargs)

            def accept_completion(buffer):
                if buffer.text == "/clear":
                    pipe.send_text("\r")

            prompt.default_buffer.on_text_changed += accept_completion
            return prompt

        read_prompt = display.read_prompt
        inputs = iter(["/cl\t", "first\r", "next\r", "/exit\r"])

        def next_prompt():
            keys = next(inputs)
            if keys == "next\r":
                model.text, model.interrupt = "answer", False
            pipe.send_text(keys)
            prompt = read_prompt()
            prompts.append(prompt)
            return prompt

        monkeypatch.setattr(display, "read_prompt", next_prompt)
        monkeypatch.setattr(display, "status", statuses.append)
        monkeypatch.setattr(terminal, "Terminal", lambda *_: display)
        monkeypatch.setattr(backend, "Backend", lambda *_: model)
        monkeypatch.setattr(prompt_toolkit, "PromptSession", make_prompt)
        monkeypatch.setattr(output, "create_output",
                            lambda **kwargs: output.DummyOutput())
        timeout = threading.Timer(5, lambda: pipe.send_text("\x03"))
        timeout.start()
        try:
            with application.create_app_session(input=pipe):
                assert cli.run(display.config, responses) == 0
        finally:
            timeout.cancel()
            timeout.join()
    assert prompts == ["/clear", "first", "next", "/exit"]
    assert any("stopped" in text.lower() for text in statuses)
    assert responses.getvalue() == "partial\nanswer\n"
    assert model.last_messages == [
        {
            "role": "user",
            "content": "first"
        },
        {
            "role": "assistant",
            "content": "partial"
        },
        {
            "role": "user",
            "content": "next"
        },
    ]
    assert model.closed and model.cancelled and model.drained
