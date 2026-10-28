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
"""Editable prompts, NVIDIA colors, and streaming terminal output."""

import contextlib
import os
import sys
import threading
import time

from prompt_toolkit import completion, document

from . import backend

BANNER = r"""
   ______   ____     ____   ______         _       _       __  __
  |  ____| |  _ \   / ___| |  ____|       | |     | |     |  \/  |
  |  __|   | | | | | |  _  |  __|    ___  | |     | |     | |\/| |
  | |____  | |_| | | |_| | | |____  |___| | |___  | |___  | |  | |
  |______| |____/   \____| |______|       |_____| |_____| |_|  |_|
"""
GREEN = "\033[38;2;118;185;0m"
DIM = "\033[2m"
RESET = "\033[0m"
PROGRESS_INTERVAL_SECONDS = 30


class PromptCompleter(completion.Completer):
    """Complete slash commands and literal file paths on Tab."""

    def __init__(self):
        self.paths = completion.PathCompleter(expanduser=True)

    def get_completions(self, doc, complete_event):
        if not doc.is_cursor_at_the_end or "\n" in doc.text:
            return
        text = doc.text
        if text.startswith("/file "):
            yield from self.paths.get_completions(document.Document(text[6:]),
                                                  complete_event)
        elif text.startswith("/"):
            for command in ("/help", "/clear", "/regen", "/file ", "/exit"):
                if command.startswith(text):
                    yield completion.Completion(command,
                                                start_position=-len(text))


class Terminal:
    """Render responses to stdout and all UI/status information to stderr."""

    def __init__(self, responses, config):
        self.responses = responses
        self.config = config
        self.interactive = all(stream.isatty()
                               for stream in (sys.stdin, responses,
                                              sys.stderr))
        color_disabled = config.no_color or "NO_COLOR" in os.environ or os.environ.get(
            "TERM") == "dumb"
        self.color = self.interactive and not color_disabled
        self._prompt = None
        self._loading_started = 0.0
        self._loading_stage = ""
        self._progress_lock = threading.Lock()
        self._response_written = False
        self._thinking = False

    def status(self, text):
        """Write one status message without contaminating response output."""
        print(text, file=sys.stderr, flush=True)

    def startup(self):
        """Show the banner and a short hint in interactive terminals."""
        if not self.interactive:
            return
        self.status((GREEN if self.color else "") + BANNER +
                    (RESET if self.color else ""))
        self.status("NVIDIA TensorRT Edge-LLM")
        self.status(
            "/help for commands  |  Ctrl+C stops a response. At the prompt, Ctrl+C or Ctrl+D exits\n"
        )

    def runtime_details(self, details):
        """Show loaded runtime details before the first interactive prompt."""
        self.status("")
        width = max((len(name) for name in details), default=0)
        for name, value in details.items():
            label = f"{name:<{width}}"
            if self.color:
                label = GREEN + label + RESET
            self.status(f"{label} : {value}")
        self.status("")

    def progress(self, stage):
        """Accept shared API loading notifications."""
        with self._progress_lock:
            self._loading_stage = stage
            self._show_progress(stage)

    def _show_progress(self, stage):
        elapsed = time.monotonic() - self._loading_started
        label = (GREEN if self.color else "") + stage + (RESET
                                                         if self.color else "")
        self.status(f"{label} [{elapsed:.1f}s]")

    @contextlib.contextmanager
    def loading(self):
        """Time loading stages without sharing a live line with native logs."""
        self._loading_started = time.monotonic()
        self._loading_stage = ""
        stopped = threading.Event()

        def refresh():
            while not stopped.wait(PROGRESS_INTERVAL_SECONDS):
                with self._progress_lock:
                    if self._loading_stage:
                        self._show_progress(self._loading_stage)

        worker = threading.Thread(target=refresh, daemon=True)
        worker.start()
        try:
            yield
        finally:
            stopped.set()
            worker.join()

    def read_prompt(self):
        """Read editable input, or one plain line when redirected."""
        if not self.interactive:
            line = sys.stdin.readline()
            if not line:
                raise EOFError
            return line.rstrip("\r\n")
        if self._prompt is None:
            import prompt_toolkit
            from prompt_toolkit import key_binding, output

            bindings = key_binding.KeyBindings()

            @bindings.add("escape", "enter")
            def newline(event):
                event.current_buffer.insert_text("\n")

            self._prompt = prompt_toolkit.PromptSession(
                [("#76b900 bold" if self.color else "", "> ")],
                key_bindings=bindings,
                completer=PromptCompleter(),
                complete_while_typing=False,
                output=output.create_output(stdout=sys.stderr),
                color_depth=output.ColorDepth.TRUE_COLOR
                if self.color else output.ColorDepth.DEPTH_1_BIT,
            )
        return self._prompt.prompt()

    def begin_response(self):
        """Reset output tracking before a response."""
        self._response_written = False

    def write(self, text):
        """Write each response delta immediately."""
        self._end_thinking()
        self._write_text(text)

    def write_reasoning(self, text):
        """Render reasoning separately from the final answer."""
        if not self._thinking:
            self._write_text("[Start thinking]\n\n" +
                             (DIM if self.color else ""))
            self._thinking = True
        self._write_text(text)

    def _end_thinking(self):
        if self._thinking:
            self._write_text((RESET if self.color else "") +
                             "\n\n[End thinking]\n\n")
            self._thinking = False

    def _write_text(self, text):
        self.responses.write(text)
        self.responses.flush()
        self._response_written = True

    def finish_response(self, session):
        """End the response and report throughput, context usage, and latency."""
        if self._response_written:
            self.write("\n")
        if session.interrupted:
            self.status("Response stopped.")
        if self.config.no_metrics or session.metrics.total_time == 0:
            return
        metrics = session.metrics
        prompt_rate, generation_rate = backend.rates(metrics)

        def rate(value):
            return f"{value:.1f}" if value is not None else "n/a"

        def tokens(value):
            return (f"{value / 1000:.2f}".rstrip("0").rstrip(".") +
                    "K") if value >= 1000 else str(value)

        capacity = session.backend.context_capacity
        context = f"n/a/{tokens(capacity)}"
        if metrics.prompt_tokens is not None:
            used = metrics.prompt_tokens + metrics.generated_tokens
            context = f"{tokens(used)}/{tokens(capacity)} ({used / capacity:.0%})"
        ttft = metrics.time_to_first_token
        first = f"{ttft:.3f}s" if ttft is not None else "n/a"
        summary = (
            f"prompt {rate(prompt_rate)} tok/s | generation {rate(generation_rate)} tok/s"
            f" | context {context} | TTFT {first} | total {metrics.total_time:.3f}s"
        )
        self.status((DIM if self.color else "") + summary +
                    (RESET if self.color else ""))
