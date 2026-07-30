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
"""OpenAI-compatible tool request validation and output parsing.

References the OpenAI-compatible tool-calling API shape documented by vLLM:
https://docs.vllm.ai/en/stable/features/tool_calling/
"""

import ast
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


@dataclass
class ToolConfig:
    tools: List[Dict[str, Any]] = field(default_factory=list)
    tool_choice: str = "none"
    forced_name: Optional[str] = None

    @property
    def parse_output(self) -> bool:
        return bool(self.tools) and self.tool_choice != "none"

    @property
    def names(self) -> set[str]:
        return {
            tool["function"]["name"]
            for tool in self.tools if isinstance(tool.get("function"), dict)
        }


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str

    def to_openai(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


@dataclass
class ParsedAssistantOutput:
    events: List[Dict[str, Any]]
    malformed: bool = False

    @property
    def content(self) -> str:
        return "".join(e["text"] for e in self.events
                       if e["type"] == "content")

    @property
    def reasoning(self) -> str:
        return "".join(e["text"] for e in self.events
                       if e["type"] == "reasoning")

    @property
    def tool_calls(self) -> List[ToolCall]:
        return [
            e["tool_call"] for e in self.events if e["type"] == "tool_call"
        ]


def validate_tool_request(
    messages: Sequence[Dict[str, Any]],
    tools: Optional[Sequence[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
) -> ToolConfig:
    """Validate OpenAI-style tool fields and message links."""
    if tools is None:
        tool_list: List[Dict[str, Any]] = []
    elif isinstance(tools, list):
        tool_list = list(tools)
    else:
        raise ValueError("'tools' must be an array")

    names = _validate_tools(tool_list)
    choice, forced_name = _validate_tool_choice(tool_choice, names, tool_list)
    _validate_tool_messages(messages)
    return ToolConfig(tools=tool_list,
                      tool_choice=choice,
                      forced_name=forced_name)


def parse_assistant_output(text: str, tool_config: ToolConfig,
                           model_dir: str) -> ParsedAssistantOutput:
    """Parse model text into ordered content, reasoning, and tool-call events."""
    if not tool_config.parse_output:
        return ParsedAssistantOutput(_split_reasoning_events(text))

    parser = _select_parser(model_dir)
    events, malformed = parser.parse(text, tool_config)
    expanded: List[Dict[str, Any]] = []
    for event in events:
        if event["type"] == "content":
            expanded.extend(_split_reasoning_events(event["text"]))
        else:
            expanded.append(event)
    return ParsedAssistantOutput(expanded, malformed=malformed)


def _validate_tools(tools: Sequence[Dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"tools[{idx}] must be an object")
        if tool.get("type") != "function":
            raise ValueError(f"tools[{idx}].type must be 'function'")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ValueError(f"tools[{idx}].function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"tools[{idx}].function.name must be a string")
        if name in names:
            raise ValueError(f"Duplicate tool name: {name}")
        names.add(name)
        desc = function.get("description")
        if desc is not None and not isinstance(desc, str):
            raise ValueError(
                f"tools[{idx}].function.description must be a string")
        params = function.get("parameters")
        if params is not None and not isinstance(params, dict):
            raise ValueError(
                f"tools[{idx}].function.parameters must be an object")
        strict = function.get("strict")
        if strict is not None and not isinstance(strict, bool):
            raise ValueError(f"tools[{idx}].function.strict must be a bool")
    return names


def _validate_tool_choice(
        tool_choice: Optional[Union[str, Dict[str, Any]]], names: set[str],
        tools: Sequence[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
    if tool_choice is None:
        return ("auto" if tools else "none"), None
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none", "required"}:
            if tool_choice in {"auto", "required"} and not tools:
                raise ValueError("'tool_choice' requires at least one tool")
            return tool_choice, None
        if tool_choice not in names:
            raise ValueError(f"Unknown forced tool name: {tool_choice}")
        return "function", tool_choice
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") != "function":
            raise ValueError("tool_choice.type must be 'function'")
        function = tool_choice.get("function")
        if not isinstance(function, dict):
            raise ValueError("tool_choice.function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool_choice.function.name must be a string")
        if name not in names:
            raise ValueError(f"Unknown forced tool name: {name}")
        return "function", name
    raise ValueError(
        "'tool_choice' must be 'auto', 'none', 'required', or a function choice"
    )


def _validate_tool_messages(messages: Sequence[Dict[str, Any]]) -> None:
    seen: set[str] = set()
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"messages[{idx}] must be an object")
        role = msg.get("role")
        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                raise ValueError(
                    f"messages[{idx}].tool_calls must be an array")
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    raise ValueError(
                        f"messages[{idx}].tool_calls entries must be objects")
                tc_id = tc.get("id")
                if not isinstance(tc_id, str) or not tc_id:
                    raise ValueError(
                        f"messages[{idx}].tool_calls[].id must be a string")
                function = tc.get("function")
                if not isinstance(function, dict):
                    raise ValueError(
                        f"messages[{idx}].tool_calls[].function must be an object"
                    )
                name = function.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError(
                        f"messages[{idx}].tool_calls[].function.name must be a string"
                    )
                seen.add(tc_id)
        elif role == "tool":
            tc_id = msg.get("tool_call_id")
            if not isinstance(tc_id, str) or not tc_id:
                raise ValueError(
                    f"messages[{idx}].tool_call_id must be a string")
            if tc_id not in seen:
                raise ValueError(f"Dangling tool_call_id: {tc_id}")
            content = msg.get("content", "")
            if content is not None and not isinstance(content,
                                                      (str, dict, list)):
                raise ValueError(
                    f"messages[{idx}].content must be text or JSON")


def _split_reasoning_events(text: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    pos = 0
    for match in re.finditer(r"<think>(.*?)</think>", text, flags=re.S):
        if match.start() > pos:
            events.append({"type": "content", "text": text[pos:match.start()]})
        events.append({"type": "reasoning", "text": match.group(1).strip()})
        pos = match.end()
    if pos < len(text):
        tail = text[pos:]
        if tail:
            events.append({"type": "content", "text": tail})
    return events or [{"type": "content", "text": ""}]


class _GenericToolParser:

    _BLOCK_RE = re.compile(
        r"(<tool_call>.*?</tool_call>|<tool_calls>.*?</tool_calls>|"
        r"<toolcall>.*?</toolcall>|<toolcalls>.*?</toolcalls>|"
        r"<function_call>.*?</function_call>|"
        r"<function_calls>.*?</function_calls>|"
        r"<function=[^>]+>.*?</function>|"
        r"\[TOOL_CALLS?\].*?(?:\[/TOOL_CALLS?\]|$))",
        re.S,
    )

    def parse(self, text: str,
              tool_config: ToolConfig) -> Tuple[List[Dict[str, Any]], bool]:
        events: List[Dict[str, Any]] = []
        malformed = False
        pos = 0
        matched = False
        for match in self._BLOCK_RE.finditer(text):
            matched = True
            if match.start() > pos:
                events.append({
                    "type": "content",
                    "text": text[pos:match.start()]
                })
            calls = _parse_tool_block(match.group(0), tool_config)
            if calls:
                events.extend({
                    "type": "tool_call",
                    "tool_call": c
                } for c in calls)
            else:
                malformed = True
                events.append({"type": "content", "text": match.group(0)})
            pos = match.end()
        if pos < len(text):
            events.append({"type": "content", "text": text[pos:]})
        if matched:
            return events, malformed

        calls = _parse_tool_block(text, tool_config)
        if calls:
            return ([{
                "type": "tool_call",
                "tool_call": c
            } for c in calls], False)
        return [{"type": "content", "text": text}], False


class _ToolParserRegistry:

    def __init__(self):
        parser = _GenericToolParser()
        self._parsers = {
            "generic": parser,
            "hermes": parser,
            "qwen3_xml": parser,
            "nemotron": parser,
            "openai": parser,
        }

    def get(self, model_dir: str):
        return self._parsers.get(_parser_name_for_model(model_dir),
                                 self._parsers["generic"])


_PARSERS = _ToolParserRegistry()


def _select_parser(model_dir: str):
    return _PARSERS.get(model_dir)


def _parser_name_for_model(model_dir: str) -> str:
    model_type = ""
    try:
        with open(os.path.join(model_dir, "config.json")) as f:
            model_type = str(json.load(f).get("model_type", "")).lower()
    except (OSError, ValueError):
        pass
    name = f"{model_type} {os.path.basename(model_dir).lower()}"
    if "qwen3" in name and "coder" in name:
        return "qwen3_xml"
    if "qwen" in name:
        return "hermes"
    if "nemotron" in name:
        return "nemotron"
    if "openai" in name or "gpt-oss" in name:
        return "openai"
    return "generic"


def _parse_tool_block(block: str, tool_config: ToolConfig) -> List[ToolCall]:
    body = _strip_tool_tags(block)
    calls = _parse_qwen_xml_calls(body, tool_config)
    if calls:
        return calls
    payload = _loads_payload(body)
    if payload is None:
        calls = _parse_pythonic_calls(body, tool_config)
        return calls
    return list(_calls_from_payload(payload, tool_config))


def _strip_tool_tags(text: str) -> str:
    body = text.strip()
    for pattern in [
            r"^<tool_call>\s*(.*?)\s*</tool_call>$",
            r"^<tool_calls>\s*(.*?)\s*</tool_calls>$",
            r"^<toolcall>\s*(.*?)\s*</toolcall>$",
            r"^<toolcalls>\s*(.*?)\s*</toolcalls>$",
            r"^<function_call>\s*(.*?)\s*</function_call>$",
            r"^<function_calls>\s*(.*?)\s*</function_calls>$",
            r"^\[TOOL_CALLS?\]\s*(.*?)\s*(?:\[/TOOL_CALLS?\])?$",
    ]:
        match = re.match(pattern, body, flags=re.S)
        if match:
            return match.group(1).strip()
    return _strip_code_fence(body)


def _strip_code_fence(text: str) -> str:
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text.strip(), flags=re.S)
    return match.group(1).strip() if match else text.strip()


def _loads_payload(text: str) -> Optional[Any]:
    body = _strip_code_fence(text)
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(body)
        except (ValueError, SyntaxError, TypeError):
            pass
    return None


def _calls_from_payload(payload: Any,
                        tool_config: ToolConfig) -> Iterable[ToolCall]:
    if isinstance(payload, dict) and isinstance(payload.get("tool_calls"),
                                                list):
        payload = payload["tool_calls"]
    if isinstance(payload, dict):
        call = _call_from_dict(payload, tool_config)
        if call:
            yield call
        return
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                call = _call_from_dict(item, tool_config)
                if call:
                    yield call


def _call_from_dict(data: Dict[str, Any],
                    tool_config: ToolConfig) -> Optional[ToolCall]:
    function = data.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = data.get("name")
        arguments = data.get("arguments", data.get("parameters", {}))
    if not isinstance(name, str) or not _tool_name_allowed(name, tool_config):
        return None
    return ToolCall(
        id=data.get("id")
        if isinstance(data.get("id"), str) else _new_call_id(),
        name=name,
        arguments=_arguments_to_json(arguments),
    )


def _parse_qwen_xml_calls(text: str,
                          tool_config: ToolConfig) -> List[ToolCall]:
    calls = []
    for match in re.finditer(r"<function=([^>]+)>(.*?)</function>",
                             text,
                             flags=re.S):
        name = match.group(1).strip()
        if not _tool_name_allowed(name, tool_config):
            continue
        parameter_schemas = _tool_parameter_schemas(name, tool_config)
        args: Dict[str, Any] = {}
        for param in re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>",
                                 match.group(2),
                                 flags=re.S):
            parameter_name = param.group(1).strip()
            args[parameter_name] = _parse_qwen_parameter_value(
                param.group(2).strip(),
                parameter_schemas.get(parameter_name, {}),
            )
        calls.append(
            ToolCall(id=_new_call_id(),
                     name=name,
                     arguments=_arguments_to_json(args)))
    return calls


def _tool_parameter_schemas(name: str,
                            tool_config: ToolConfig) -> Dict[str, Any]:
    for tool in tool_config.tools:
        function = tool.get("function")
        if not isinstance(function, dict) or function.get("name") != name:
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            return {}
        properties = parameters.get("properties")
        return properties if isinstance(properties, dict) else {}
    return {}


def _parse_qwen_parameter_value(value: Any, schema: Dict[str, Any]) -> Any:
    """Restore XML parameter types using the request's JSON schema."""
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        parameter_types = [schema_type]
    elif isinstance(schema_type, list):
        if not schema_type or not all(
                isinstance(item, str) for item in schema_type):
            return value
        parameter_types = list(dict.fromkeys(schema_type))
    else:
        return value

    if "null" in parameter_types:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "null":
            return None

    non_null_types = [item for item in parameter_types if item != "null"]
    if len(non_null_types) != 1:
        return value
    parameter_type = non_null_types[0]

    parsed = value
    if isinstance(value, str):
        if parameter_type == "string":
            return value
        parsed = _loads_payload(value)
        if parsed is None:
            return value

    if parameter_type == "integer":
        return parsed if isinstance(
            parsed, int) and not isinstance(parsed, bool) else value
    if parameter_type == "number":
        return parsed if isinstance(
            parsed, (int, float)) and not isinstance(parsed, bool) else value
    if parameter_type == "boolean":
        return parsed if isinstance(parsed, bool) else value
    if parameter_type == "array" and isinstance(parsed, list):
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return parsed
        return [
            _parse_qwen_parameter_value(item, item_schema) for item in parsed
        ]
    if parameter_type == "object" and isinstance(parsed, dict):
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return parsed
        return {
            key: _parse_qwen_parameter_value(item, properties.get(key, {}))
            for key, item in parsed.items()
        }
    return value


def _parse_pythonic_calls(text: str,
                          tool_config: ToolConfig) -> List[ToolCall]:
    calls = []
    for line in text.strip().splitlines():
        match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*,?\s*$", line)
        if not match:
            continue
        name = match.group(1)
        if not _tool_name_allowed(name, tool_config):
            continue
        calls.append(
            ToolCall(id=_new_call_id(),
                     name=name,
                     arguments=_arguments_to_json(
                         _parse_python_args(match.group(2)))))
    return calls


def _parse_python_args(args_src: str) -> Dict[str, Any]:
    try:
        expr = ast.parse(f"f({args_src})", mode="eval").body
    except SyntaxError:
        return {"__raw__": args_src}
    if not isinstance(expr, ast.Call):
        return {"__raw__": args_src}
    args: Dict[str, Any] = {}
    for kw in expr.keywords:
        if kw.arg is not None:
            try:
                args[kw.arg] = ast.literal_eval(kw.value)
            except ValueError:
                args[kw.arg] = ast.unparse(kw.value)
    return args


def _tool_name_allowed(name: str, tool_config: ToolConfig) -> bool:
    if tool_config.forced_name and name != tool_config.forced_name:
        return False
    names = tool_config.names
    return not names or name in names


def _arguments_to_json(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            json.loads(arguments)
            return arguments
        except ValueError:
            return json.dumps({"__raw__": arguments}, ensure_ascii=False)
    return json.dumps(arguments or {}, ensure_ascii=False)


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"
