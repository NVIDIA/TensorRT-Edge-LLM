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

from experimental.server.tool_chat_template import (ToolChatTemplateFormatter,
                                                    needs_tool_chat_template)


def test_needs_tool_template():
    assert needs_tool_chat_template([{
        "role": "user",
        "content": "hi"
    }],
                                    tools=[{
                                        "type": "function"
                                    }])
    assert needs_tool_chat_template([{
        "role": "user",
        "content": "hi"
    }],
                                    tool_choice="required")
    assert needs_tool_chat_template([{
        "role": "assistant",
        "tool_calls": []
    }, {
        "role": "tool",
        "content": "42"
    }])
    assert not needs_tool_chat_template([{"role": "user", "content": "hi"}])


class _RecordingTemplateOwner:

    def __init__(self):
        self.kwargs = None
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return json.dumps(
            {
                "messages": messages,
                "tools": kwargs["tools"],
                "tool_choice": kwargs.get("tool_choice"),
                "add_generation_prompt": kwargs["add_generation_prompt"],
            },
            sort_keys=True)


class _ReplayTemplateOwner:

    def __init__(self, extra_generation_token=False):
        self.extra_generation_token = extra_generation_token
        self.tokenizer = self
        self.apply_count = 0

    def apply_chat_template(self, messages, **kwargs):
        self.apply_count += 1
        prefix = "history<assistant><think>\n"
        if kwargs["add_generation_prompt"]:
            suffix = "\n</think>\n\n"
            if self.extra_generation_token:
                suffix += "<extra>"
            return prefix + suffix
        assistant = messages[-1]
        return (prefix + assistant["reasoning_content"] + "\n</think>\n\n" +
                assistant["content"])

    def encode(self, prompt, *, add_special_tokens):
        assert add_special_tokens is False
        special_token_ids = {
            "</think>": 256,
            "<extra>": 257,
            "\n\n": 258,
        }
        token_ids = []
        offset = 0
        while offset < len(prompt):
            special_token = next((token for token in special_token_ids
                                  if prompt.startswith(token, offset)), None)
            if special_token is None:
                token_ids.append(ord(prompt[offset]))
                offset += 1
            else:
                token_ids.append(special_token_ids[special_token])
                offset += len(special_token)
        return token_ids


def test_formats_tool_template():
    owner = _RecordingTemplateOwner()
    formatter = ToolChatTemplateFormatter([], template_owner=owner)
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object"
            },
        },
    }]
    prompt = formatter.format(
        [{
            "role":
            "assistant",
            "content":
            None,
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"city\":\"Paris\"}",
                },
            }],
        }, {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "get_weather",
            "content": {
                "temperature": 22
            },
        }],
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {
                "name": "get_weather"
            },
        },
    )

    formatted = json.loads(prompt)
    tool_call = formatted["messages"][0]["tool_calls"][0]
    tool_message = formatted["messages"][1]
    assert formatted["tools"] == tools
    assert formatted["tool_choice"] == {
        "type": "function",
        "function": {
            "name": "get_weather"
        },
    }
    assert formatted["add_generation_prompt"] is True
    assert tool_call["function"]["arguments"] == {"city": "Paris"}
    assert tool_message["content"] == '{"temperature": 22}'


def test_computes_context_reuse_replay_tail_from_tokenized_template():
    owner = _ReplayTemplateOwner()
    formatter = ToolChatTemplateFormatter([], template_owner=owner)
    _, replay_tail_length = formatter.format_with_replay_tail([{
        "role":
        "user",
        "content":
        "hello",
    }])
    assert replay_tail_length == 4
    _, replay_tail_length = formatter.format_with_replay_tail([{
        "role":
        "user",
        "content":
        "another request",
    }])
    assert replay_tail_length == 4
    assert owner.apply_count == 3

    formatter = ToolChatTemplateFormatter(
        [], template_owner=_ReplayTemplateOwner(extra_generation_token=True))
    _, replay_tail_length = formatter.format_with_replay_tail([{
        "role":
        "user",
        "content":
        "hello",
    }])
    assert replay_tail_length == 5
