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

import json

import pytest

from experimental.server.parsing.tool_calling import (parse_assistant_output,
                                                      stream_assistant_output,
                                                      validate_tool_request)


def _tools():
    return [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string"
                    }
                },
            },
        },
    }]


def _config(tool_choice="auto", parallel=True):
    return validate_tool_request(
        [{
            "role": "user",
            "content": "Weather?"
        }],
        _tools(),
        tool_choice,
        parallel,
    )


def test_validates_tool_request():
    messages = [{
        "role":
        "assistant",
        "content":
        None,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city":"Paris"}',
            },
        }],
    }, {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": {
            "temperature": 22
        },
    }]
    config = validate_tool_request(messages, _tools(), "required", False)
    assert config.tool_choice == "required"
    assert not config.parallel_tool_calls

    with pytest.raises(ValueError, match="Unknown forced tool name"):
        validate_tool_request(messages, _tools(), {
            "type": "function",
            "function": {
                "name": "missing"
            },
        })
    with pytest.raises(ValueError, match="Dangling tool_call_id"):
        validate_tool_request([{
            "role": "tool",
            "tool_call_id": "missing",
            "content": "42",
        }])


def test_parses_reasoning_and_multiple_tool_calls(tmp_path):
    text = (
        "<think>plan</think>Before"
        "<function=get_weather><parameter=city>Paris</parameter></function>"
        '<tool_call>{"name":"get_weather","arguments":{"city":"Tokyo"}}'
        "</tool_call>After")
    parsed = parse_assistant_output(text,
                                    _config(),
                                    str(tmp_path),
                                    reasoning_parser="qwen3")
    assert parsed.reasoning == "plan"
    assert parsed.content == "BeforeAfter"
    assert [json.loads(call.arguments)["city"]
            for call in parsed.tool_calls] == ["Paris", "Tokyo"]


def test_normal_output_is_content_without_reasoning_parser(tmp_path):
    parsed = parse_assistant_output("ordinary answer", _config("none"),
                                    str(tmp_path))
    assert parsed.content == "ordinary answer"
    assert parsed.reasoning == ""


def test_filters_forced_tool(tmp_path):
    text = '<tool_call>{"name":"other","arguments":{}}</tool_call>'
    parsed = parse_assistant_output(
        text,
        _config({
            "type": "function",
            "function": {
                "name": "get_weather"
            },
        }),
        str(tmp_path),
    )
    assert parsed.tool_calls == []
    assert parsed.content == text


def test_streams_content_reasoning_and_tools_across_chunk_boundaries(tmp_path):
    parser = stream_assistant_output(_config(),
                                     str(tmp_path),
                                     reasoning_parser="qwen3")
    events = []
    for chunk in (
            "<th",
            "ink>plan</think>Before<tool_",
            'call>{"name":"get_weather","arguments":{"city":',
            '"Paris"}}</tool_call>After',
    ):
        events.extend(parser.feed(chunk))
    events.extend(parser.flush())

    assert "".join(event["text"] for event in events
                   if event["type"] == "reasoning") == "plan"
    assert "".join(event["text"] for event in events
                   if event["type"] == "content") == "BeforeAfter"
    calls = [
        event["tool_call"] for event in events if event["type"] == "tool_call"
    ]
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert json.loads(calls[0].arguments) == {"city": "Paris"}


def test_stream_parser_flushes_untagged_provider_format(tmp_path):
    parser = stream_assistant_output(_config(), str(tmp_path))
    events = list(parser.feed('get_weather(city="Paris")'))
    events.extend(parser.flush())

    assert len(events) == 1
    assert events[0]["type"] == "tool_call"
    assert json.loads(events[0]["tool_call"].arguments) == {"city": "Paris"}
