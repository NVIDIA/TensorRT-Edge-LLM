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
"""Tests for the OpenAI Responses compatibility layer used by Codex."""

import json
import os
import sys

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tensorrt_edgellm.server.api.responses_compat import (ResponsesStream,
                                                          to_chat_request,
                                                          to_response_object,
                                                          to_sse)


def test_converts_codex_turn_to_chat():
    body = {
        "model":
        "qwen35-local",
        "instructions":
        "be terse",
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": [{
                    "type": "input_text",
                    "text": "today is Thursday"
                }],
            },
            {
                "type": "message",
                "role": "user",
                "content": "Reply ready.",
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "exec_command",
                "arguments": "{\"cmd\":\"true\"}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok",
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "exec_command",
                "description": "run",
                "parameters": {
                    "type": "object"
                },
            },
            {
                "type": "web_search",
                "external_web_access": False
            },
            {
                "type":
                "namespace",
                "name":
                "agents",
                "tools": [{
                    "type": "function",
                    "name": "spawn_agent",
                    "description": "spawn",
                    "parameters": {
                        "type": "object"
                    },
                }],
            },
        ],
        "tool_choice":
        "auto",
        "max_output_tokens":
        32,
    }
    chat = to_chat_request(body)
    assert chat["messages"][0] == {
        "role": "system",
        "content": "be terse\n\ntoday is Thursday",
    }
    assert chat["messages"][1] == {"role": "user", "content": "Reply ready."}
    assert chat["messages"][2]["tool_calls"][0]["function"]["name"] == (
        "exec_command")
    assert chat["messages"][3] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "ok",
    }
    assert [tool["function"]["name"]
            for tool in chat["tools"]] == ["exec_command", "spawn_agent"]
    assert chat["max_tokens"] == 32
    assert chat["tool_choice"] == "auto"


def test_keeps_input_images_on_messages_and_tool_results():
    body = {
        "model":
        "qwen35-local",
        "input": [
            {
                "type":
                "message",
                "role":
                "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "what is this"
                    },
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,aaaa",
                        "detail": "auto",
                    },
                ],
            },
            {
                "type":
                "function_call_output",
                "call_id":
                "call_img",
                "output": [
                    {
                        "type": "input_text",
                        "text": "screenshot"
                    },
                    {
                        "type": "input_image",
                        "image_url": {
                            "url": "https://example.com/a.png"
                        },
                    },
                ],
            },
        ],
    }
    chat = to_chat_request(body)
    assert chat["messages"][0]["content"] == [
        {
            "type": "text",
            "text": "what is this"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,aaaa",
                "detail": "auto",
            },
        },
    ]
    assert chat["messages"][1]["content"] == [
        {
            "type": "text",
            "text": "screenshot"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/a.png"
            },
        },
    ]


def test_emits_responses_stream_for_text_and_tool_call():
    response = {
        "model":
        "qwen35-local",
        "choices": [{
            "message": {
                "role":
                "assistant",
                "content":
                "ready",
                "tool_calls": [{
                    "id": "call_9",
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "arguments": "{\"cmd\":\"true\"}",
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 4
        },
    }
    completed = to_response_object(response)
    assert completed["status"] == "completed"
    assert completed["output"][0]["content"][0]["text"] == "ready"
    assert completed["output"][1]["call_id"] == "call_9"
    assert completed["usage"]["input_tokens"] == 12
    stream = to_sse(response)
    assert "event: response.output_text.delta" in stream
    assert "event: response.function_call_arguments.done" in stream
    assert "event: response.completed" in stream
    payload = stream.strip().split("data: ")[-1]
    assert json.loads(payload)["response"]["status"] == "completed"


def _chat_sse(delta, finish_reason=None):
    payload = {
        "id":
        "chatcmpl-1",
        "model":
        "qwen35-local",
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    return "data: " + json.dumps(payload) + "\n\n"


def test_streams_text_and_tool_arguments_as_they_arrive():
    stream = ResponsesStream("qwen35-local")
    opened = "".join(stream.start())
    assert "event: response.created" in opened
    assert "response.completed" not in opened

    first = "".join(stream.push_sse(_chat_sse({"content": "Rea"})))
    assert "event: response.output_text.delta" in first
    assert '"delta": "Rea"' in first
    assert "response.completed" not in first

    second = "".join(stream.push_sse(_chat_sse({"content": "dy"})))
    assert '"delta": "dy"' in second
    assert "response.completed" not in second

    call = "".join(
        stream.push_sse(
            _chat_sse({
                "tool_calls": [{
                    "index": 0,
                    "id": "call_9",
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "arguments": "",
                    },
                }]
            })))
    assert "event: response.output_text.done" in call
    assert "event: response.output_item.added" in call
    assert "response.completed" not in call

    args = "".join(
        stream.push_sse(
            _chat_sse({
                "tool_calls": [{
                    "index": 0,
                    "function": {
                        "arguments": "{\"cmd\":\"true\"}"
                    },
                }]
            })))
    assert "event: response.function_call_arguments.delta" in args
    assert "response.completed" not in args

    done = "".join(stream.push_sse(_chat_sse({}, finish_reason="tool_calls")))
    assert "event: response.function_call_arguments.done" in done
    assert "event: response.completed" in done
    assert "Ready" in done
