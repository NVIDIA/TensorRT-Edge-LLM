# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Agentic chat: device telemetry, the bounded tool loop, and its settings."""

import json
import signal
import subprocess
import types

import pytest

from experimental.frontend import agent, session
from tensorrt_edgellm import server
from tensorrt_edgellm.server.parsing import reasoning, tool_calling
from tensorrt_edgellm.server.runtime import engine

from . import helpers

# Dummy tegrastats samples; the values are illustrative.
GR3D_SAMPLE = ("RAM 3000/7620MB (lfb 500x4MB) CPU [15%@729,10%@729] "
               "GR3D_FREQ 12%@[1098,1098,1098] CPU@53.5C GPU@45.5C")
IGPU_SAMPLE = ("RAM 52125/59669MB (lfb 96x4MB) CPU [12%,8%] IGPU_FREQ 37% "
               "cpu@42.5C/42.5C gpu@41.8C/41.8C soc345@40.5C/40.5C")


@pytest.mark.parametrize("line,utilization,temperature",
                         [(IGPU_SAMPLE, 37.0, 41.8), (GR3D_SAMPLE, 12.0, 45.5),
                          ("RAM 1/2MB", None, None)])
def test_tegrastats_reports_shared_memory_load_and_temperature(
        line, utilization, temperature):
    """Tegra samples map to the tool fields; absent readings become null."""
    status = agent.parse_tegrastats(line)
    assert status["utilization_percent"] == utilization
    assert status["temperature_c"] == temperature
    assert status["memory_total_mib"] > status["memory_used_mib"] > 0
    assert status["unified_memory"]
    with pytest.raises(ValueError):
        agent.parse_tegrastats("WARNING: Run tegrastats as root")


def test_unused_huge_pages_are_reserved_not_used():
    """A mostly empty huge-page pool counts as reserved, not as usage."""
    meminfo = ("MemTotal:       61100636 kB\nHugePages_Total:   24576\n"
               "HugePages_Free:    24543\nHugepagesize:       2048 kB\n")
    status = agent.parse_tegrastats(IGPU_SAMPLE, meminfo)
    assert status["memory_reserved_mib"] == 24543 * 2
    assert status["memory_used_mib"] == 52125 - 24543 * 2
    assert agent.parse_tegrastats(IGPU_SAMPLE)["memory_reserved_mib"] == 0


def test_nvidia_smi_fields_and_failures_are_tool_results(monkeypatch):
    """nvidia-smi queries the CUDA device, [N/A] becomes null, and failures
    stay data."""
    commands = []

    def run(command, **_):
        commands.append(command)
        return types.SimpleNamespace(
            stdout="Example GPU, 512, 2048, [N/A], 41\n")

    monkeypatch.setattr(agent.shutil, "which", lambda name: name)
    monkeypatch.setattr(agent, "_cuda_pci_bus_id", lambda: "0000:65:00.0")
    monkeypatch.setattr(agent.subprocess, "run", run)
    assert json.loads(agent.call("device_status")) == {
        "device": "Example GPU",
        "memory_used_mib": 512.0,
        "memory_total_mib": 2048.0,
        "utilization_percent": None,
        "temperature_c": 41.0,
        "unified_memory": False,
    }
    assert "--id=0000:65:00.0" in commands[0]

    def timeout(*_, **__):
        raise subprocess.TimeoutExpired("nvidia-smi", agent.TIMEOUT_SECONDS)

    monkeypatch.setattr(agent.subprocess, "run", timeout)
    assert "error" in json.loads(agent.call("device_status"))
    assert "error" in json.loads(agent.call("rm_rf"))


class ToolBackend(helpers.FakeBackend):
    """Call device_status for the first tool_turns offers, otherwise answer."""

    def __init__(self, tool_turns=1):
        super().__init__()
        self.tool_turns = tool_turns
        self.requests = []

    def stream(self, messages, on_metrics, tools=None):
        self.requests.append((messages, tools))
        calls = sum(1 for _, offered in self.requests if offered)
        item = (tool_calling.ToolCall(f"call_{len(self.requests)}",
                                      "device_status", "{}") if tools
                and calls <= self.tool_turns else reasoning.ReasoningDelta(
                    "content", "GPU is idle."))

        def iterate():
            try:
                yield item
            finally:
                on_metrics(engine.GenerationMetrics(total_time=0.5))

        return server.CancellableIterator(iterate(), lambda: None)


@pytest.mark.parametrize("tool_turns,rounds", [(1, 1),
                                               (100, agent.MAX_TOOL_ROUNDS)])
def test_tool_loop_records_results_and_is_bounded(monkeypatch, tool_turns,
                                                  rounds):
    """Tool results precede the answer; the last round offers no tools, asks
    for an answer, and the turn's total time spans every round."""
    model = ToolBackend(tool_turns)
    monkeypatch.setattr(agent, "call", lambda name: '{"temperature_c":40}')
    monkeypatch.setattr(session.time, "monotonic",
                        iter([100.0, 107.5, 200.0, 201.0]).__next__)
    shown = []
    conversation = session.Session(model, [], agentic=True)
    conversation.generate("How busy is my GPU?",
                          lambda _: None,
                          show_tool=lambda *call: shown.append(call))
    history = conversation.history
    assert [message["role"] for message in history
            ] == ["user"] + ["assistant", "tool"] * rounds + ["assistant"]
    assert history[1]["tool_calls"][0]["function"]["arguments"] == {}
    assert history[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"temperature_c":40}'
    }
    assert history[-1]["content"] == "GPU is idle."
    assert shown == [("device_status", '{"temperature_c":40}')] * rounds
    assert model.requests[0][1] == agent.TOOLS
    final_request = model.requests[-1][0]
    assert ((final_request[-1] == agent.FINAL_ROUND_MESSAGE) == (
        rounds == agent.MAX_TOOL_ROUNDS))
    assert agent.FINAL_ROUND_MESSAGE not in history
    assert conversation.metrics.total_time == 7.5

    sent = len(model.requests)
    conversation.regenerate(lambda _: None)
    assert model.requests[sent][0] == history[:1]
    assert conversation.history[-1]["content"] == "GPU is idle."


def test_failed_later_round_reports_no_metrics(monkeypatch):
    """A turn that fails after a tool round shows no stale timing summary."""
    model = ToolBackend()
    validate = model.validate

    def reject_after_tools(messages, tools=None):
        if messages[-1]["role"] == "tool":
            raise ValueError("Conversation too long")
        return validate(messages, tools)

    model.validate = reject_after_tools
    monkeypatch.setattr(agent, "call", lambda name: "{}")
    conversation = session.Session(model, [], agentic=True)
    with pytest.raises(ValueError, match="too long"):
        conversation.generate("How busy?", lambda _: None)
    assert conversation.metrics.total_time == 0
    assert conversation.history == []


def test_cancel_during_tools_skips_remaining_calls_and_history(monkeypatch):
    """Ctrl+C during a tool call skips later calls and keeps history intact."""
    model = ToolBackend()

    def stream(messages, on_metrics, tools=None):
        model.requests.append((messages, tools))
        calls = (tool_calling.ToolCall(f"call_{index}", "device_status", "{}")
                 for index in range(2))
        return server.CancellableIterator(calls, lambda: None)

    def interrupted_call(name):
        signal.raise_signal(signal.SIGINT)
        return "{}"

    model.stream = stream
    monkeypatch.setattr(agent, "call", interrupted_call)
    shown = []
    conversation = session.Session(model, [], agentic=True)
    conversation.generate("How busy?",
                          lambda _: None,
                          show_tool=lambda *call: shown.append(call))
    assert conversation.interrupted and len(shown) == 1
    assert len(model.requests) == 1 and conversation.history == []


def test_plain_chat_offers_no_tools():
    """Without --agentic the session never offers tools to the model."""
    model = ToolBackend()
    conversation = session.Session(model, [])
    conversation.generate("hi", lambda _: None)
    assert model.requests == [([{"role": "user", "content": "hi"}], None)]


@pytest.mark.parametrize("system", [None, "Be brief."])
def test_agentic_extends_system_prompt_and_rejects_guided_output(
        tmp_path, system):
    """--agentic appends tool instructions and refuses guided decoding."""
    options = ["--agentic"] + (["--system", system] if system else [])
    settings = helpers.settings(tmp_path, *options)
    assert settings.agentic
    assert settings.messages == [{
        "role":
        "system",
        "content": (system + "\n\n" if system else "") + agent.SYSTEM_PROMPT
    }]
    assert helpers.settings(tmp_path).messages == []
    schema = tmp_path / "schema.json"
    schema.write_text("{}")
    with pytest.raises(ValueError, match="guided decoding"):
        helpers.settings(tmp_path, "--agentic", "--json-schema", str(schema))
