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
import json
import signal
import time

from experimental.frontend import agent
from tensorrt_edgellm.server.parsing import tool_calling
from tensorrt_edgellm.server.runtime import engine

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

    def __init__(self, backend, messages, agentic=False):
        self.backend = backend
        self.agentic = agentic
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

    def generate(self, prompt, write, write_reasoning=None, show_tool=None):
        """Stream a turn on the main thread; SIGINT cancels and drains it."""
        messages = self.history + [{"role": "user", "content": prompt}]
        self._generate(messages, write, write_reasoning, show_tool)

    def regenerate(self, write, write_reasoning=None, show_tool=None):
        """Replace the latest response, committing history only after generation."""
        self.metrics = engine.GenerationMetrics()
        self.interrupted = False
        prompts = [
            index for index, message in enumerate(self.history) if
            index >= len(self.initial_messages) and message["role"] == "user"
        ]
        if not prompts or self.history[-1]["role"] != "assistant":
            raise ValueError("No response to regenerate. Send a prompt first.")
        self._generate(self.history[:prompts[-1] + 1], write, write_reasoning,
                       show_tool)

    def _call_tools(self, calls, visible, reasoning, show_tool):
        """Return the assistant tool-call message followed by each tool result."""
        request = {"role": "assistant", "content": "".join(visible)}
        if reasoning:
            request["reasoning_content"] = "".join(reasoning)
        request["tool_calls"] = []
        results = []
        for call in calls:
            if self.interrupted:
                break
            try:
                arguments = json.loads(call.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            # Object arguments render through chat templates that iterate them.
            request["tool_calls"].append({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": arguments
                },
            })
            result = agent.call(call.name)
            if show_tool:
                show_tool(call.name, result)
            results.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result
            })
        return [request] + results

    def _generate(self, messages, write, write_reasoning, show_tool):
        if self.state != "ready":
            raise RuntimeError(
                f"Cannot generate while session is {self.state}")
        self.metrics = engine.GenerationMetrics()
        self.interrupted = False
        self.state = "generating"
        visible = []
        reasoning = []
        completed = False
        started = time.monotonic()
        tool_round = 0
        previous_handler = signal.signal(signal.SIGINT, self._cancel)
        try:
            for tool_round in range(agent.MAX_TOOL_ROUNDS + 1):
                tools = None
                request = messages
                if tool_round < agent.MAX_TOOL_ROUNDS:
                    tools = agent.TOOLS if self.agentic else None
                else:
                    # Not committed to history; makes the last round answer.
                    request = messages + [agent.FINAL_ROUND_MESSAGE]
                visible = []
                reasoning = []
                calls = []
                # A round that fails before streaming leaves no stale metrics.
                self.metrics = engine.GenerationMetrics()
                self.metrics.prompt_tokens = self.backend.validate(
                    request, tools)
                if self.interrupted:
                    return
                self._stream = self.backend.stream(request, self._set_metrics,
                                                   tools)
                if self.interrupted:
                    self._stream.cancel()
                for delta in self._stream:
                    if self.interrupted:
                        break
                    if isinstance(delta, tool_calling.ToolCall):
                        calls.append(delta)
                    elif delta.text:
                        if getattr(delta, "field", "content") == "reasoning":
                            (write_reasoning or write)(delta.text)
                            reasoning.append(delta.text)
                        else:
                            write(delta.text)
                            visible.append(delta.text)
                self._stream.close()
                self._stream = None
                if self.interrupted or not calls:
                    break
                messages = messages + self._call_tools(calls, visible,
                                                       reasoning, show_tool)
            if tool_round:
                self.metrics.total_time = time.monotonic() - started
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
