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
"""Pure translations between Anthropic Messages and OpenAI chat shapes."""

import json
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "stop_sequence": "stop_sequence",
}


def _blocks_to_text(content: Any) -> str:
    """Flatten text blocks while ignoring non-text protocol content."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(block.get("text", "")) for block in content
        if isinstance(block, dict) and block.get("type") == "text")


def _image_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """Translate one Anthropic image source to OpenAI image content."""
    source = block.get("source")
    if not isinstance(source, dict):
        raise ValueError("image.source must be an object")
    source_type = source.get("type")
    if source_type == "base64":
        media_type = source.get("media_type")
        data = source.get("data")
        if not isinstance(media_type, str) or not isinstance(data, str):
            raise ValueError(
                "base64 image sources require media_type and data strings")
        url = f"data:{media_type};base64,{data}"
    elif source_type == "url" and isinstance(source.get("url"), str):
        url = source["url"]
    else:
        raise ValueError("image.source.type must be 'base64' or 'url'")
    return {"type": "image_url", "image_url": {"url": url}}


def _openai_content(blocks: List[Dict[str, Any]]) -> Any:
    """Use plain text when no typed multimodal blocks are required."""
    if blocks and all(block.get("type") == "text" for block in blocks):
        return "\n".join(block["text"] for block in blocks)
    return blocks


def _tool_call_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """Translate one client tool invocation."""
    return {
        "id": block.get("id", ""),
        "type": "function",
        "function": {
            "name": block.get("name", ""),
            "arguments": json.dumps(block.get("input") or {},
                                    ensure_ascii=False),
        },
    }


def _tool_result_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """Translate one tool result and retain its failure status in-band."""
    result = block.get("content", "")
    if not isinstance(result, str):
        result = _blocks_to_text(result) or json.dumps(result,
                                                       ensure_ascii=False)
    if block.get("is_error"):
        result = f"[tool error] {result}"
    return {
        "role": "tool",
        "tool_call_id": block.get("tool_use_id", ""),
        "content": result,
    }


def _convert_content_block(block: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Classify and translate one typed Anthropic content block."""
    if not isinstance(block, dict):
        raise ValueError("message content blocks must be objects")
    block_type = block.get("type")
    if block_type == "text":
        return "content", {
            "type": "text",
            "text": str(block.get("text", "")),
        }
    if block_type == "image":
        return "content", _image_block(block)
    if block_type == "tool_use":
        return "tool_call", _tool_call_block(block)
    if block_type == "tool_result":
        return "tool_result", _tool_result_block(block)
    if block_type in {"thinking", "redacted_thinking"}:
        return "ignored", None
    raise ValueError(f"unsupported Anthropic content block: {block_type}")


def _convert_content_blocks(
    content: List[Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition Anthropic content into chat content, calls, and results."""
    content_blocks: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    for block in content:
        kind, converted = _convert_content_block(block)
        if converted is None:
            continue
        if kind == "content":
            content_blocks.append(converted)
        elif kind == "tool_call":
            tool_calls.append(converted)
        elif kind == "tool_result":
            tool_results.append(converted)
    return content_blocks, tool_calls, tool_results


def _convert_message(message: Any, index: int) -> List[Dict[str, Any]]:
    """Translate one Anthropic user or assistant message."""
    if not isinstance(message, dict):
        raise ValueError(f"messages[{index}] must be an object")
    role = message.get("role")
    if role not in {"user", "assistant"}:
        raise ValueError(f"messages[{index}].role must be user or assistant")
    content = message.get("content", "")
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        raise ValueError(f"messages[{index}].content must be text or an array")

    content_blocks, tool_calls, tool_results = _convert_content_blocks(content)
    if role == "assistant":
        assistant: Dict[str, Any] = {
            "role": "assistant",
            "content": _openai_content(content_blocks) or None,
        }
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        return [assistant]

    converted = list(tool_results)
    if content_blocks or not tool_results:
        converted.append({
            "role": "user",
            "content": _openai_content(content_blocks),
        })
    return converted


def _convert_messages(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Translate the system prompt and ordered conversation messages."""
    messages: List[Dict[str, Any]] = []
    system = _blocks_to_text(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})

    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("messages must be a non-empty array")
    for index, message in enumerate(raw_messages):
        messages.extend(_convert_message(message, index))
    return messages


def _convert_tools(raw_tools: Any) -> Optional[List[Dict[str, Any]]]:
    """Translate client tools and omit unsupported server-executed tools."""
    if raw_tools is None:
        return None
    if not isinstance(raw_tools, list):
        raise ValueError("tools must be an array")
    tools = []
    for tool in raw_tools:
        if not isinstance(tool, dict):
            raise ValueError("tool definitions must be objects")
        schema = tool.get("input_schema")
        if not isinstance(schema, dict):
            continue
        tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": schema,
            },
        })
    return tools


def _convert_tool_choice(raw_choice: Any) -> Optional[Any]:
    """Map Anthropic tool-selection modes to the shared chat contract."""
    if raw_choice is None:
        return None
    if not isinstance(raw_choice, dict):
        raise ValueError("tool_choice must be an object")
    choice_type = raw_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "tool":
        return {
            "type": "function",
            "function": {
                "name": raw_choice.get("name", "")
            },
        }
    if choice_type == "none":
        return "none"
    raise ValueError("unsupported Anthropic tool_choice.type")


def _sampling_options(body: Dict[str, Any],
                      require_max_tokens: bool) -> Dict[str, Any]:
    """Validate and translate generation controls."""
    max_tokens = body.get("max_tokens")
    if max_tokens is None and not require_max_tokens:
        max_tokens = 1
    if isinstance(max_tokens,
                  bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    thinking = body.get("thinking")
    return {
        "max_tokens":
        max_tokens,
        "temperature":
        body.get("temperature", 0.7),
        "top_p":
        body.get("top_p", 0.9),
        "top_k":
        body.get("top_k", 50),
        "stop":
        list(body.get("stop_sequences") or []),
        "enable_thinking": (isinstance(thinking, dict)
                            and thinking.get("type") == "enabled"),
    }


def convert_request(
    body: Dict[str, Any],
    *,
    require_max_tokens: bool = True,
) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Optional[Any],
           Dict[str, Any]]:
    """Convert an Anthropic Messages request to the internal chat contract."""
    return (
        _convert_messages(body),
        _convert_tools(body.get("tools")),
        _convert_tool_choice(body.get("tool_choice")),
        _sampling_options(body, require_max_tokens),
    )


def convert_stop_reason(finish_reason: Optional[str]) -> str:
    """Map an internal finish reason to an Anthropic stop reason."""
    return _STOP_REASON_MAP.get(finish_reason or "stop", "end_turn")


def build_content_blocks(
    content: Optional[str],
    tool_calls: List[Dict[str, Any]],
    reasoning: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build Anthropic response blocks from generated text and tool calls."""
    blocks: List[Dict[str, Any]] = []
    if reasoning:
        blocks.append({
            "type": "thinking",
            "thinking": reasoning,
            "signature": "",
        })
    if content:
        blocks.append({"type": "text", "text": content})
    for call in tool_calls:
        function = call.get("function", {})
        try:
            tool_input = json.loads(function.get("arguments") or "{}")
        except (TypeError, ValueError):
            tool_input = {"__raw__": function.get("arguments")}
        if not isinstance(tool_input, dict):
            tool_input = {"value": tool_input}
        blocks.append({
            "type": "tool_use",
            "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:16]}",
            "name": function.get("name", ""),
            "input": tool_input,
        })
    return blocks or [{"type": "text", "text": ""}]


def usage(prompt_tokens: Optional[int],
          completion_tokens: int) -> Dict[str, int]:
    """Build an Anthropic usage object from exact runtime token counts."""
    return {
        "input_tokens": prompt_tokens or 0,
        "output_tokens": completion_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def event(event_type: str, payload: Dict[str, Any]) -> str:
    """Serialize one Anthropic server-sent event."""
    value = dict(payload)
    value["type"] = event_type
    return (f"event: {event_type}\n"
            f"data: {json.dumps(value, ensure_ascii=False)}\n\n")


def message_start_events(message_id: str, model: str,
                         prompt_tokens: Optional[int]) -> Iterable[str]:
    """Yield the required opening events for a Messages stream."""
    yield event(
        "message_start", {
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": usage(prompt_tokens, 0),
            }
        })
    yield event("ping", {})


def content_tail_events(blocks: List[Dict[str, Any]], stop_reason: str,
                        completion_tokens: int) -> Iterable[str]:
    """Yield ordered content blocks and closing Messages stream events."""
    for index, block in enumerate(blocks):
        block_type = block["type"]
        if block_type == "text":
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        elif block_type == "thinking":
            start = {"type": "thinking", "thinking": "", "signature": ""}
            delta = {
                "type": "thinking_delta",
                "thinking": block["thinking"],
            }
        else:
            start = {
                "type": "tool_use",
                "id": block["id"],
                "name": block["name"],
                "input": {},
            }
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block["input"], ensure_ascii=False),
            }
        yield event("content_block_start", {
            "index": index,
            "content_block": start,
        })
        yield event("content_block_delta", {"index": index, "delta": delta})
        yield event("content_block_stop", {"index": index})
    yield event(
        "message_delta", {
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None
            },
            "usage": {
                "output_tokens": completion_tokens
            },
        })
    yield event("message_stop", {})


def error_payload(status: int, message: str) -> Dict[str, Any]:
    """Build an Anthropic error object for an HTTP status."""
    if status == 413:
        error_type = "request_too_large"
    elif status == 429:
        error_type = "rate_limit_error"
    elif status == 529:
        error_type = "overloaded_error"
    elif status < 500:
        error_type = "invalid_request_error"
    else:
        error_type = "api_error"
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message
        },
    }
