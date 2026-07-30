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

from experimental.server.api_server import (_build_message_body,
                                            _generate_stream_sse)
from experimental.server.engine import StreamDelta
from experimental.server.tool_calling import (parse_assistant_output,
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
                    },
                },
            },
        },
    }]


def _tool_config(tool_choice="auto"):
    return validate_tool_request([{
        "role": "user",
        "content": "Weather?"
    }], _tools(), tool_choice)


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
                "arguments": "{\"city\":\"Paris\"}",
            },
        }],
    }, {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": {
            "temperature": 22
        },
    }]

    config = validate_tool_request(messages, _tools(), "auto")

    assert config.tool_choice == "auto"

    with pytest.raises(ValueError, match="Unknown forced tool name"):
        validate_tool_request([{
            "role": "user",
            "content": "hi"
        }], _tools(), {
            "type": "function",
            "function": {
                "name": "missing"
            },
        })

    with pytest.raises(ValueError, match="'tools' must be an array"):
        validate_tool_request([{
            "role": "user",
            "content": "hi"
        }], {"type": "function"})

    with pytest.raises(ValueError, match="Dangling tool_call_id"):
        validate_tool_request([{
            "role": "tool",
            "tool_call_id": "call_missing",
            "content": "42",
        }])


def test_parses_tool_calls(tmp_path):
    json_text = ("Let me check.\n"
                 "<tool_call>{\"name\":\"get_weather\","
                 "\"arguments\":{\"city\":\"Paris\"}}</tool_call>")

    parsed = parse_assistant_output(json_text, _tool_config(), str(tmp_path))

    assert parsed.content == "Let me check.\n"
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "get_weather"
    assert json.loads(parsed.tool_calls[0].arguments) == {"city": "Paris"}

    qwen_text = (
        "<think>plan</think>Before"
        "<function=get_weather><parameter=city>Paris</parameter></function>"
        "After")

    parsed = parse_assistant_output(qwen_text, _tool_config(), str(tmp_path))

    assert [event["type"] for event in parsed.events
            ] == ["reasoning", "content", "tool_call", "content"]
    assert parsed.reasoning == "plan"
    assert parsed.content == "BeforeAfter"
    assert json.loads(parsed.tool_calls[0].arguments) == {"city": "Paris"}


def test_qwen_xml_parser_restores_json_schema_types(tmp_path):
    tools = [{
        "type": "function",
        "function": {
            "name": "submit_job",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer"
                    },
                    "ratio": {
                        "type": "number"
                    },
                    "enabled": {
                        "type": "boolean"
                    },
                    "label": {
                        "type": "string"
                    },
                    "ids": {
                        "type": "array",
                        "items": {
                            "type": "integer"
                        },
                    },
                    "options": {
                        "type": "object",
                        "properties": {
                            "retries": {
                                "type": "integer"
                            },
                            "code": {
                                "type": "string"
                            },
                        },
                    },
                },
            },
        },
    }]
    config = validate_tool_request([{
        "role": "user",
        "content": "Submit"
    }], tools, "auto")
    text = (
        "<tool_call><function=submit_job>"
        "<parameter=count>3</parameter>"
        "<parameter=ratio>0.5</parameter>"
        "<parameter=enabled>true</parameter>"
        "<parameter=label>007</parameter>"
        "<parameter=ids>[\"1\", 2]</parameter>"
        "<parameter=options>{\"retries\": \"4\", \"code\": \"009\"}</parameter>"
        "</function></tool_call>")

    parsed = parse_assistant_output(text, config, str(tmp_path))

    assert json.loads(parsed.tool_calls[0].arguments) == {
        "count": 3,
        "ratio": 0.5,
        "enabled": True,
        "label": "007",
        "ids": [1, 2],
        "options": {
            "retries": 4,
            "code": "009"
        },
    }


@pytest.mark.parametrize(
    "schema,xml_value,expected",
    [
        ({
            "type": ["integer", "null"]
        }, "5", 5),
        ({
            "type": ["integer", "null"]
        }, "null", None),
        ({
            "type": ["string", "null"]
        }, "007", "007"),
        ({
            "type": ["string", "null"]
        }, "None", "None"),
        ({
            "type": ["string", "null"]
        }, "NULL", "NULL"),
        ({
            "type": ["array", "null"],
            "items": {
                "type": "integer"
            },
        }, '["1", 2]', [1, 2]),
        ({
            "type": ["object", "null"],
            "properties": {
                "retries": {
                    "type": "integer"
                },
                "code": {
                    "type": "string"
                },
            },
        }, '{"retries": "4", "code": "009"}', {
            "retries": 4,
            "code": "009"
        }),
        ({
            "type": ["integer", "number", "null"]
        }, "5", "5"),
        ({
            "type": ["boolean", "string", "null"]
        }, "true", "true"),
        ({
            "type": ["integer", 7, "null"]
        }, "5", "5"),
    ],
)
def test_qwen_xml_parser_handles_schema_unions(tmp_path, schema, xml_value,
                                               expected):
    tools = [{
        "type": "function",
        "function": {
            "name": "submit_value",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": schema
                },
            },
        },
    }]
    config = validate_tool_request([{
        "role": "user",
        "content": "Submit"
    }], tools, "auto")
    text = ("<tool_call><function=submit_value>"
            f"<parameter=value>{xml_value}</parameter>"
            "</function></tool_call>")

    parsed = parse_assistant_output(text, config, str(tmp_path))

    assert json.loads(parsed.tool_calls[0].arguments) == {"value": expected}


def test_filters_forced_tool(tmp_path):
    text = "<tool_call>{\"name\":\"other\",\"arguments\":{}}</tool_call>"
    parsed = parse_assistant_output(
        text,
        _tool_config({
            "type": "function",
            "function": {
                "name": "get_weather"
            },
        }),
        str(tmp_path),
    )

    assert parsed.tool_calls == []
    assert parsed.content == text


class _FakeLLM:

    def __init__(self, model_dir):
        self.model_dir = str(model_dir)

    def generate_stream(self,
                        messages,
                        params,
                        *,
                        tools=None,
                        tool_choice=None):
        yield StreamDelta(text="<think>plan</think>", finished=False)
        yield StreamDelta(
            text="<tool_call>{\"name\":\"get_weather\","
            "\"arguments\":{\"city\":\"Paris\"}}</tool_call>",
            finished=True,
            finish_reason="stop",
        )


def test_builds_tool_response_shapes(tmp_path):
    config = _tool_config()
    message, has_tool_calls = _build_message_body(
        "<tool_call>{\"name\":\"get_weather\","
        "\"arguments\":{\"city\":\"Paris\"}}</tool_call>",
        config,
        str(tmp_path),
    )

    assert has_tool_calls
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "get_weather"
    chunks = list(
        _generate_stream_sse(
            _FakeLLM(tmp_path),
            [{
                "role": "user",
                "content": "Weather?"
            }],
            object(),
            "chatcmpl-test",
            False,
            tool_config=config,
        ))
    payloads = [
        json.loads(chunk.removeprefix("data: ")) for chunk in chunks
        if chunk.startswith("data: {")
    ]

    assert payloads[1]["choices"][0]["delta"]["reasoning"] == "plan"
    assert "tool_calls" in payloads[2]["choices"][0]["delta"]
    assert payloads[3]["choices"][0]["delta"]["tool_calls"][0]["function"][
        "arguments"] == '{"city": "Paris"}'
    assert payloads[-1]["choices"][0]["finish_reason"] == "tool_calls"
