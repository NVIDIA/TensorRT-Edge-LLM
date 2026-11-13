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
"""Translate OpenAI Responses requests onto the chat-completions runtime.

Codex CLI speaks only ``POST /v1/responses``. This module keeps that protocol
at the HTTP edge and reuses the existing chat path for generation.
"""

import json
import uuid
from typing import Any, Dict, List, Optional

_TEXT_BLOCK_TYPES = {"input_text", "output_text", "text"}
_IMAGE_BLOCK_TYPES = {"input_image", "image_url", "image"}
_SYSTEM_ROLES = {"system", "developer"}


def _image_part(block: Dict[str, Any]) -> Dict[str, Any]:
    """Translate one Responses image block into a chat image_url part."""
    image = block.get("image_url", block.get("image"))
    url = image.get("url") if isinstance(image, dict) else image
    if not isinstance(url, str) or not url:
        if block.get("file_id"):
            raise ValueError(
                "input_image file_id is not supported; pass image_url")
        raise ValueError("input_image requires image_url")
    image_url: Dict[str, Any] = {"url": url}
    detail = block.get("detail")
    if not isinstance(detail, str) and isinstance(image, dict):
        detail = image.get("detail")
    if isinstance(detail, str) and detail:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def _chat_content(content: Any) -> Any:
    """Keep text as a string, and keep images as chat content parts."""
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        return ""
    parts: List[Dict[str, Any]] = []
    saw_image = False
    for block in content:
        if isinstance(block, str):
            if block:
                parts.append({"type": "text", "text": block})
            continue
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind in _TEXT_BLOCK_TYPES:
            text = str(block.get("text") or "")
            if text:
                parts.append({"type": "text", "text": text})
            continue
        if kind in _IMAGE_BLOCK_TYPES:
            parts.append(_image_part(block))
            saw_image = True
    if not saw_image:
        return "\n".join(part["text"] for part in parts)
    return parts


def _text_from_content(content: Any) -> str:
    converted = _chat_content(content)
    if isinstance(converted, str):
        return converted
    return "\n".join(part["text"] for part in converted
                     if part.get("type") == "text")


def _function_tools(tools: Any) -> List[Dict[str, Any]]:
    """Keep client function tools and drop server-executed tools such as web search."""
    if not isinstance(tools, list):
        return []
    converted: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        kind = tool.get("type")
        if kind == "namespace":
            converted.extend(_function_tools(tool.get("tools")))
            continue
        if kind != "function" or not isinstance(tool.get("name"), str):
            continue
        parameters = tool.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}
        converted.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description") or "",
                "parameters": parameters,
            },
        })
    return converted


def _tool_output(output: Any) -> Any:
    """Keep image blocks in a tool result; leave other structured output as JSON."""
    if isinstance(output, str) or output is None:
        return output or ""
    blocks = output if isinstance(output, list) else [output]
    content_types = _TEXT_BLOCK_TYPES | _IMAGE_BLOCK_TYPES
    recognized = [
        block for block in blocks if isinstance(block, str) or (
            isinstance(block, dict) and block.get("type") in content_types)
    ]
    if recognized and len(recognized) == len(blocks):
        return _chat_content(blocks)
    return json.dumps(output, ensure_ascii=False)


def _append_function_calls(pending: List[Dict[str, Any]],
                           messages: List[Dict[str, Any]]) -> None:
    if not pending:
        return
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": list(pending),
    })
    pending.clear()


def to_chat_request(body: Dict[str, Any]) -> Dict[str, Any]:
    """Build a chat-completions body from one Responses request."""
    if not isinstance(body, dict):
        raise ValueError("request body must be an object")
    system_parts: List[str] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        system_parts.append(instructions)

    raw_input = body.get("input")
    if isinstance(raw_input, str):
        items: List[Any] = [{
            "type": "message",
            "role": "user",
            "content": raw_input
        }]
    elif isinstance(raw_input, list):
        items = raw_input
    else:
        raise ValueError("input must be a string or an array")

    messages: List[Dict[str, Any]] = []
    pending_calls: List[Dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"input[{index}] must be an object")
        kind = item.get("type") or "message"
        if kind == "message":
            _append_function_calls(pending_calls, messages)
            role = item.get("role") or "user"
            content = _chat_content(item.get("content"))
            if role in _SYSTEM_ROLES:
                text = _text_from_content(content)
                if text:
                    system_parts.append(text)
                if isinstance(content, list):
                    messages.append({"role": "user", "content": content})
                continue
            if role not in {"user", "assistant"}:
                raise ValueError(
                    f"input[{index}] has unsupported role {role!r}")
            messages.append({"role": role, "content": content})
            continue
        if kind == "function_call":
            name = item.get("name")
            call_id = item.get("call_id") or item.get("id")
            if not isinstance(name, str) or not isinstance(call_id, str):
                raise ValueError(
                    f"input[{index}] function_call requires name and call_id")
            arguments = item.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {}, ensure_ascii=False)
            pending_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments
                },
            })
            continue
        if kind == "function_call_output":
            _append_function_calls(pending_calls, messages)
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError(f"input[{index}] requires call_id")
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": _tool_output(item.get("output")),
            })
            continue
        raise ValueError(f"input[{index}] has unsupported type {kind!r}")
    _append_function_calls(pending_calls, messages)

    chat_messages: List[Dict[str, Any]] = []
    if system_parts:
        chat_messages.append({
            "role": "system",
            "content": "\n\n".join(system_parts),
        })
    chat_messages.extend(messages)
    if not chat_messages:
        raise ValueError("input must contain at least one message")

    request: Dict[str, Any] = {
        "messages": chat_messages,
        "model": body.get("model"),
    }
    tools = _function_tools(body.get("tools"))
    if tools:
        request["tools"] = tools
        choice = body.get("tool_choice")
        if choice is not None:
            request["tool_choice"] = choice
        if "parallel_tool_calls" in body:
            request["parallel_tool_calls"] = bool(
                body.get("parallel_tool_calls"))
    max_output = body.get("max_output_tokens")
    if isinstance(max_output, int):
        request["max_tokens"] = max_output
    for field in ("temperature", "top_p"):
        if field in body and body[field] is not None:
            request[field] = body[field]
    return request


def _usage(response: Dict[str, Any]) -> Optional[Dict[str, int]]:
    usage = response.get("usage") or {}
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if prompt is None and completion is None:
        return None
    prompt = int(prompt or 0)
    completion = int(completion or 0)
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": prompt + completion,
    }


def to_response_object(response: Dict[str, Any]) -> Dict[str, Any]:
    """Turn one chat completion into a Responses object."""
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    output: List[Dict[str, Any]] = []
    if text:
        output.append({
            "id":
            "msg_" + uuid.uuid4().hex,
            "type":
            "message",
            "role":
            "assistant",
            "status":
            "completed",
            "content": [{
                "type": "output_text",
                "text": text,
                "annotations": [],
            }],
        })
    for call in tool_calls:
        function = call.get("function") or {}
        output.append({
            "id": "fc_" + uuid.uuid4().hex,
            "type": "function_call",
            "status": "completed",
            "call_id": call.get("id") or "",
            "name": function.get("name") or "",
            "arguments": function.get("arguments") or "",
        })
    completed: Dict[str, Any] = {
        "id": "resp_" + uuid.uuid4().hex,
        "object": "response",
        "status": "completed",
        "model": response.get("model"),
        "output": output,
    }
    usage = _usage(response)
    if usage is not None:
        completed["usage"] = usage
    return completed


def _event(name: str, data: Dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class ResponsesStream:
    """Turn chat-completion SSE into Responses events as each delta arrives."""

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model
        self.resp_id = "resp_" + uuid.uuid4().hex
        self.output: List[Dict[str, Any]] = []
        self._text: Optional[Dict[str, Any]] = None
        self._tools: Dict[int, Dict[str, Any]] = {}
        self._next_index = 0
        self._usage: Optional[Dict[str, int]] = None
        self._finished = False
        self._failed = False

    def start(self) -> List[str]:
        return [
            _event(
                "response.created", {
                    "type": "response.created",
                    "response": self._snapshot("in_progress"),
                })
        ]

    def push_sse(self, chunk: str) -> List[str]:
        events: List[str] = []
        for line in chunk.splitlines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):].strip()
            if data == "[DONE]":
                events.extend(self.finish())
                continue
            events.extend(self.push(json.loads(data)))
        return events

    def push(self, payload: Dict[str, Any]) -> List[str]:
        if self._failed or self._finished:
            return []
        if payload.get("error") and not payload.get("choices"):
            self._failed = True
            return [
                _event("error", {
                    "type": "error",
                    "error": payload["error"],
                })
            ]
        if payload.get("model"):
            self.model = payload["model"]
        usage = _usage(payload)
        if usage is not None:
            self._usage = usage
        choice = (payload.get("choices") or [None])[0]
        if not isinstance(choice, dict):
            return []
        delta = choice.get("delta") or {}
        events: List[str] = []
        content = delta.get("content")
        if content:
            events.extend(self._ensure_text())
            assert self._text is not None
            self._text["text"] += content
            events.append(
                _event(
                    "response.output_text.delta", {
                        "type": "response.output_text.delta",
                        "item_id": self._text["id"],
                        "output_index": self._text["output_index"],
                        "content_index": 0,
                        "delta": content,
                    }))
        for call in delta.get("tool_calls") or []:
            events.extend(self._tool_delta(call))
        if choice.get("finish_reason"):
            events.extend(self.finish())
        return events

    def finish(self) -> List[str]:
        if self._finished or self._failed:
            return []
        self._finished = True
        events = self._close_text()
        for record in self._tools.values():
            events.extend(self._close_tool(record))
        events.append(
            _event(
                "response.completed", {
                    "type": "response.completed",
                    "response": self._snapshot("completed"),
                }))
        return events

    def _ensure_text(self) -> List[str]:
        if self._text is not None:
            return []
        events: List[str] = []
        for record in list(self._tools.values()):
            if not record.get("closed"):
                events.extend(self._close_tool(record))
        item_id = "msg_" + uuid.uuid4().hex
        output_index = self._next_index
        self._next_index += 1
        self._text = {"id": item_id, "output_index": output_index, "text": ""}
        events.append(
            _event(
                "response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {
                        "id": item_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                }))
        events.append(
            _event(
                "response.content_part.added", {
                    "type": "response.content_part.added",
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": "",
                        "annotations": [],
                    },
                }))
        return events

    def _close_text(self) -> List[str]:
        if self._text is None:
            return []
        item_id = self._text["id"]
        output_index = self._text["output_index"]
        text = self._text["text"]
        part = {"type": "output_text", "text": text, "annotations": []}
        item = {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [part],
        }
        self.output.append(item)
        self._text = None
        return [
            _event(
                "response.output_text.done", {
                    "type": "response.output_text.done",
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "text": text,
                }),
            _event(
                "response.content_part.done", {
                    "type": "response.content_part.done",
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": part,
                }),
            _event(
                "response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": item,
                }),
        ]

    def _tool_delta(self, call: Dict[str, Any]) -> List[str]:
        index = int(call.get("index") or 0)
        function = call.get("function") or {}
        events: List[str] = []
        record = self._tools.get(index)
        if record is None:
            events.extend(self._close_text())
            item_id = "fc_" + uuid.uuid4().hex
            output_index = self._next_index
            self._next_index += 1
            record = {
                "id": item_id,
                "output_index": output_index,
                "call_id": call.get("id") or "",
                "name": function.get("name") or "",
                "arguments": "",
                "closed": False,
            }
            self._tools[index] = record
            events.append(
                _event(
                    "response.output_item.added", {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": {
                            "id": item_id,
                            "type": "function_call",
                            "status": "in_progress",
                            "call_id": record["call_id"],
                            "name": record["name"],
                            "arguments": "",
                        },
                    }))
        arguments = function.get("arguments")
        if arguments:
            record["arguments"] += arguments
            events.append(
                _event(
                    "response.function_call_arguments.delta", {
                        "type": "response.function_call_arguments.delta",
                        "item_id": record["id"],
                        "output_index": record["output_index"],
                        "delta": arguments,
                    }))
        return events

    def _close_tool(self, record: Dict[str, Any]) -> List[str]:
        if record.get("closed"):
            return []
        record["closed"] = True
        item = {
            "id": record["id"],
            "type": "function_call",
            "status": "completed",
            "call_id": record["call_id"],
            "name": record["name"],
            "arguments": record["arguments"],
        }
        self.output.append(item)
        return [
            _event(
                "response.function_call_arguments.done", {
                    "type": "response.function_call_arguments.done",
                    "item_id": record["id"],
                    "output_index": record["output_index"],
                    "arguments": record["arguments"],
                }),
            _event(
                "response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": record["output_index"],
                    "item": item,
                }),
        ]

    def _snapshot(self, status: str) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "id": self.resp_id,
            "object": "response",
            "status": status,
            "model": self.model,
            "output": list(self.output),
        }
        if self._usage is not None:
            body["usage"] = self._usage
        return body


def to_sse(response: Dict[str, Any]) -> str:
    """Encode a completed Responses object as the SSE stream Codex reads."""
    completed = to_response_object(response)
    events: List[str] = []

    def emit(event: str, data: Dict[str, Any]) -> None:
        events.append(
            f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        )

    in_progress = dict(completed)
    in_progress["status"] = "in_progress"
    in_progress["output"] = []
    emit("response.created", {
        "type": "response.created",
        "response": in_progress,
    })
    for index, item in enumerate(completed["output"]):
        started = dict(item)
        started["status"] = "in_progress"
        if item["type"] == "message":
            started["content"] = []
            emit(
                "response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": started,
                })
            text = item["content"][0]["text"]
            emit(
                "response.content_part.added", {
                    "type": "response.content_part.added",
                    "item_id": item["id"],
                    "output_index": index,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": "",
                        "annotations": [],
                    },
                })
            emit(
                "response.output_text.delta", {
                    "type": "response.output_text.delta",
                    "item_id": item["id"],
                    "output_index": index,
                    "content_index": 0,
                    "delta": text,
                })
            emit(
                "response.output_text.done", {
                    "type": "response.output_text.done",
                    "item_id": item["id"],
                    "output_index": index,
                    "content_index": 0,
                    "text": text,
                })
            emit(
                "response.content_part.done", {
                    "type": "response.content_part.done",
                    "item_id": item["id"],
                    "output_index": index,
                    "content_index": 0,
                    "part": item["content"][0],
                })
        else:
            started["arguments"] = ""
            emit(
                "response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": started,
                })
            emit(
                "response.function_call_arguments.delta", {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item["id"],
                    "output_index": index,
                    "delta": item.get("arguments") or "",
                })
            emit(
                "response.function_call_arguments.done", {
                    "type": "response.function_call_arguments.done",
                    "item_id": item["id"],
                    "output_index": index,
                    "arguments": item.get("arguments") or "",
                })
        emit(
            "response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": index,
                "item": item,
            })
    emit("response.completed", {
        "type": "response.completed",
        "response": completed,
    })
    return "".join(events)
