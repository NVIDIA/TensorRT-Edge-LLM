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
"""Tests for the Anthropic Messages API compatibility layer."""

import json
import os
import sys
from itertools import chain

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experimental.server.api.anthropic_compat import (build_content_blocks,
                                                      content_tail_events,
                                                      convert_request,
                                                      convert_stop_reason,
                                                      message_start_events)


def test_converts_request():
    body = {
        "system": [{
            "type": "text",
            "text": "be terse",
            "cache_control": {
                "type": "ephemeral"
            }
        }],
        "messages": [
            {
                "role": "user",
                "content": "weather in paris?"
            },
            {
                "role":
                "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "checking"
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {
                            "city": "Paris"
                        }
                    },
                ]
            },
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{
                            "type": "text",
                            "text": "21C sunny"
                        }]
                    },
                ]
            },
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "d",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string"
                        }
                    }
                }
            },
            {
                "name": "web_search",
                "type": "web_search_20250305"
            },
        ],
        "tool_choice": {
            "type": "any"
        },
        "max_tokens":
        64,
        "stop_sequences": ["\n\nHuman:"],
    }
    messages, tools, tool_choice, sampling = convert_request(body)

    assert messages[0] == {"role": "system", "content": "be terse"}
    assert messages[1] == {"role": "user", "content": "weather in paris?"}
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "checking"
    assert messages[2]["tool_calls"][0]["id"] == "toolu_1"
    assert json.loads(
        messages[2]["tool_calls"][0]["function"]["arguments"]) == {
            "city": "Paris"
        }
    assert messages[3] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "21C sunny"
    }
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "get_weather"
    assert tool_choice == "required"
    assert sampling["stop"] == ["\n\nHuman:"]


def test_flags_tool_result_error():
    # is_error tool_result content is prefixed so the model sees it failed.
    # (Server-tool skipping is covered by test_converts_request.)
    body = {
        "messages": [{
            "role":
            "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "boom",
                    "is_error": True
                },
            ]
        }],
        "max_tokens":
        8
    }
    messages, _, _, _ = convert_request(body)
    assert messages[0]["content"] == "[tool error] boom"


def test_maps_stop_reasons():
    assert convert_stop_reason("stop") == "end_turn"
    assert convert_stop_reason("length") == "max_tokens"
    assert convert_stop_reason("tool_calls") == "tool_use"
    assert convert_stop_reason(None) == "end_turn"


def test_builds_content_blocks():
    blocks = build_content_blocks("hi", [{
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "f",
            "arguments": '{"a": 1}'
        },
    }])
    assert blocks[0] == {"type": "text", "text": "hi"}
    assert blocks[1]["type"] == "tool_use"
    assert blocks[1]["input"] == {"a": 1}
    # Non-object arguments are wrapped: tool_use.input must be an object.
    wrapped = build_content_blocks(None, [{
        "id": "call_2",
        "type": "function",
        "function": {
            "name": "g",
            "arguments": "[1, 2]"
        },
    }])
    assert wrapped[0]["input"] == {"value": [1, 2]}
    # Empty content -> an empty text block (an empty content array makes Claude
    # Code treat the stream as truncated); a tool-only response omits it.
    assert build_content_blocks("", []) == [{"type": "text", "text": ""}]
    assert [b["type"] for b in wrapped] == ["tool_use"]


def test_stream_event_grammar():
    blocks = build_content_blocks("hello", [{
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "f",
            "arguments": "{}"
        },
    }])
    events = []
    raw_events = chain(message_start_events("msg_1", "m", 10),
                       content_tail_events(blocks, "tool_use", 5))
    for raw in raw_events:
        etype, data = raw.split("\n", 1)
        events.append((etype.removeprefix("event: "),
                       json.loads(data.removeprefix("data: "))))

    types = [e[0] for e in events]
    assert types[0] == "message_start"
    assert types[-2:] == ["message_delta", "message_stop"]
    # Blocks open/close sequentially and never interleave.
    open_idx = None
    for etype, data in events:
        if etype == "content_block_start":
            assert open_idx is None
            open_idx = data["index"]
        elif etype == "content_block_delta":
            assert data["index"] == open_idx
        elif etype == "content_block_stop":
            assert data["index"] == open_idx
            open_idx = None
    assert open_idx is None
    assert events[0][1]["message"]["usage"]["input_tokens"] == 10
    assert events[-2][1]["delta"]["stop_reason"] == "tool_use"
    assert events[-2][1]["usage"]["output_tokens"] == 5
