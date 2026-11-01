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
"""Conversation history and serial turn lifecycle."""

import copy
import signal

from experimental.server.runtime import engine

HELP = """/help        Show commands
/clear       Reset the conversation to its initial messages
/regen       Replace the last assistant response
/file PATH   Send a UTF-8 text file as the next prompt
/exit        Exit

Enter sends. Alt+Enter adds a line. Paste can contain multiple lines.
Tab completes slash commands and /file paths.
Ctrl+C stops a response. Ctrl+C at the prompt or Ctrl+D exits."""


class Session:
    """Run one turn at a time and retain only completed or visible responses."""

    def __init__(self, backend, messages):
        self.backend = backend
        self.initial_messages = copy.deepcopy(messages)
        self.history = copy.deepcopy(messages)
        self.state = "ready"
        self.metrics = engine.GenerationMetrics()
        self.interrupted = False
        self._stream = None

    def command(self, text):
        """Handle lifecycle commands; return status text or None for a prompt."""
        if text == "/help":
            return HELP
        if text == "/clear":
            self.history = copy.deepcopy(self.initial_messages)
            return "Conversation cleared."
        if text == "/exit":
            self.close()
            return "Goodbye."
        return None

    def _cancel(self, _signum, _frame):
        self.interrupted = True
        self.state = "cancelling"
        if self._stream is not None:
            self._stream.cancel()

    def _set_metrics(self, metrics):
        if metrics.prompt_tokens is None:
            metrics.prompt_tokens = self.metrics.prompt_tokens
        self.metrics = metrics

    def generate(self, prompt, write, write_reasoning=None):
        """Stream a turn on the main thread; SIGINT cancels and drains it."""
        messages = self.history + [{"role": "user", "content": prompt}]
        self._generate(messages, write, write_reasoning)

    def regenerate(self, write, write_reasoning=None):
        """Replace the latest response, committing history only after generation."""
        self.metrics = engine.GenerationMetrics()
        self.interrupted = False
        if (len(self.history) < len(self.initial_messages) + 2
                or self.history[-1]["role"] != "assistant"
                or self.history[-2]["role"] != "user"):
            raise ValueError("No response to regenerate. Send a prompt first.")
        self._generate(self.history[:-1], write, write_reasoning)

    def _generate(self, messages, write, write_reasoning):
        if self.state != "ready":
            raise RuntimeError(
                f"Cannot generate while session is {self.state}")
        self.metrics = engine.GenerationMetrics()
        self.interrupted = False
        self.state = "generating"
        visible = []
        reasoning = []
        completed = False
        previous_handler = signal.signal(signal.SIGINT, self._cancel)
        try:
            self.metrics.prompt_tokens = self.backend.validate(messages)
            if self.interrupted:
                return
            self._stream = self.backend.stream(messages, self._set_metrics)
            if self.interrupted:
                self._stream.cancel()
            for delta in self._stream:
                if self.interrupted:
                    break
                if delta.text:
                    if getattr(delta, "field", "content") == "reasoning":
                        (write_reasoning or write)(delta.text)
                        reasoning.append(delta.text)
                    else:
                        write(delta.text)
                        visible.append(delta.text)
            completed = not self.interrupted
        finally:
            # A second Ctrl+C must not interrupt native worker cleanup.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                if self._stream is not None:
                    self._stream.close()
                if completed or (self.interrupted and (visible or reasoning)):
                    response = {
                        "role": "assistant",
                        "content": "".join(visible)
                    }
                    if reasoning:
                        response["reasoning_content"] = "".join(reasoning)
                    self.history = messages + [response]
            finally:
                self._stream = None
                self.state = "ready"
                signal.signal(signal.SIGINT, previous_handler)

    def close(self):
        """Stop generation and close the model."""
        if self.state != "closed":
            try:
                if self._stream is not None:
                    self._stream.close()
                self.backend.close()
            finally:
                self.state = "closed"
