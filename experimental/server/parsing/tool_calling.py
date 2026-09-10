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

The public data structures follow the OpenAI tool-calling API shape.
"""

import ast
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .reasoning import REASONING_PARSERS

logger = logging.getLogger("edgellm.server.parsing")


@dataclass
class ToolConfig:
    tools: List[Dict[str, Any]] = field(default_factory=list)
    tool_choice: str = "none"
    forced_name: Optional[str] = None
    parallel_tool_calls: bool = True

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
    parallel_tool_calls: bool = True,
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
    if not isinstance(parallel_tool_calls, bool):
        raise ValueError("'parallel_tool_calls' must be a boolean")
    return ToolConfig(tools=tool_list,
                      tool_choice=choice,
                      forced_name=forced_name,
                      parallel_tool_calls=parallel_tool_calls)


def parse_assistant_output(
        text: str,
        tool_config: ToolConfig,
        model_dir: str,
        tool_parser: str = "auto",
        reasoning_parser: str = "none") -> ParsedAssistantOutput:
    """Parse model text into ordered content, reasoning, and tool-call events."""
    if not tool_config.parse_output:
        return ParsedAssistantOutput(
            _split_reasoning_events(text,
                                    reasoning_parser,
                                    model_dir,
                                    allow_implicit=True))

    parser = _select_parser(model_dir, tool_parser)
    events, malformed = parser.parse(text, tool_config)
    expanded: List[Dict[str, Any]] = []
    for event in events:
        if event["type"] == "content":
            expanded.extend(
                _split_reasoning_events(event["text"],
                                        reasoning_parser,
                                        model_dir,
                                        allow_implicit=False))
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


def _split_reasoning_events(text: str, parser_name: str, model_dir: str, *,
                            allow_implicit: bool) -> List[Dict[str, Any]]:
    parser = REASONING_PARSERS.resolve(parser_name, model_dir)
    if parser is None:
        return [{"type": "content", "text": text}]
    if (not allow_implicit and parser.start_token not in text
            and parser.end_token not in text):
        return [{"type": "content", "text": text}]
    reasoning, content = parser.extract(text)
    events: List[Dict[str, Any]] = []
    if reasoning:
        events.append({"type": "reasoning", "text": reasoning})
    if content:
        events.append({"type": "content", "text": content})
    return events or [{"type": "content", "text": ""}]


@dataclass(frozen=True)
class _MarkerFamily:
    """One tool-call wrapper format understood by the generic parser.

    Single source of truth for the whole-text regexes derived below and for
    the streaming parser's marker matching; deriving both from this table
    keeps the non-streaming and streaming paths from drifting apart.
    """

    opens: Tuple[str, ...]
    closes: Tuple[str, ...]
    open_is_prefix: bool = False  # Open token extends to the next '>' (<function=NAME>).
    close_optional: bool = False  # End of output also terminates the block.
    strippable: bool = True  # Participates in _strip_tool_tags.


_MARKER_FAMILIES: Tuple[_MarkerFamily, ...] = (
    _MarkerFamily(("<tool_call>", ), ("</tool_call>", )),
    _MarkerFamily(("<tool_calls>", ), ("</tool_calls>", )),
    _MarkerFamily(("<toolcall>", ), ("</toolcall>", )),
    _MarkerFamily(("<toolcalls>", ), ("</toolcalls>", )),
    _MarkerFamily(("<function_call>", ), ("</function_call>", )),
    _MarkerFamily(("<function_calls>", ), ("</function_calls>", )),
    _MarkerFamily(("<function=", ), ("</function>", ),
                  open_is_prefix=True,
                  strippable=False),
    _MarkerFamily(("[TOOL_CALL]", "[TOOL_CALLS]"),
                  ("[/TOOL_CALL]", "[/TOOL_CALLS]"),
                  close_optional=True),
)


def _alternation(tokens: Sequence[str]) -> str:
    escaped = [re.escape(token) for token in tokens]
    return escaped[0] if len(escaped) == 1 else "(?:" + "|".join(escaped) + ")"


def _family_block_regex(family: _MarkerFamily) -> str:
    open_re = (re.escape(family.opens[0]) + r"[^>]+>"
               if family.open_is_prefix else _alternation(family.opens))
    close_alternatives = [re.escape(token) for token in family.closes]
    if family.close_optional:
        close_alternatives.append("$")
    close_re = (close_alternatives[0] if len(close_alternatives) == 1 else
                "(?:" + "|".join(close_alternatives) + ")")
    return open_re + ".*?" + close_re


def _family_strip_regex(family: _MarkerFamily) -> re.Pattern:
    close_re = _alternation(family.closes)
    if family.close_optional:
        close_re = "(?:" + close_re + ")?"
    return re.compile(
        "^" + _alternation(family.opens) + r"\s*(.*?)\s*" + close_re + "$",
        re.S)


_STRIP_TAG_PATTERNS = tuple(
    _family_strip_regex(family) for family in _MARKER_FAMILIES
    if family.strippable)

_OPEN_TOKENS: Tuple[str, ...] = tuple(token for family in _MARKER_FAMILIES
                                      for token in family.opens)


def _longest_partial_token(buffer: str, tokens: Sequence[str]) -> int:
    """Length of the longest buffer suffix that is a proper prefix of a token.

    That many trailing characters may still grow into a full marker and must
    be held back; a complete-token suffix returns 0 (finding complete markers
    is the marker search's job, not hold-back's).
    """
    limit = min(len(buffer), max(len(token) for token in tokens) - 1)
    for length in range(limit, 0, -1):
        tail = buffer[len(buffer) - length:]
        for token in tokens:
            if length < len(token) and token.startswith(tail):
                return length
    return 0


_JSON_WS = " \t\n\r"
_ESCAPABLE = '"\\/bfnrtu'
_LITERALS = {"t": "true", "f": "false", "n": "null"}


class _JsonScanner:
    """Char-by-char strict scanner for one JSON container value (RAW mode).

    Finds where the arguments value ends (wrapper bytes must not leak into
    the emitted arguments) and validates the strict-JSON dialect. ``push``
    returns "ok", "done" (char closes the top-level container), or "error".
    ``confirmed`` flips at the first key's value start -- the earliest point
    the dialect is known; waiting for a complete key-value pair would defer
    a single long-string argument to the end.
    """

    def __init__(self) -> None:
        self._stack: List[str] = []
        self._state = "value"
        self._string_is_key = False
        self._escape = False
        self._hex_left = 0
        self._literal = ""
        self._literal_pos = 0
        self._num_state = ""
        self.confirmed = False
        self.done = False

    def push(self, ch: str) -> str:
        if self.done:
            return "error"
        if self._state == "in_string":
            return self._push_string(ch)
        if self._state == "in_number":
            return self._push_number(ch)
        if self._state == "in_literal":
            expected = self._literal[self._literal_pos]
            if ch != expected:
                return "error"
            self._literal_pos += 1
            if self._literal_pos == len(self._literal):
                self._state = "after_value"
            return "ok"
        return self._push_structural(ch)

    def _push_string(self, ch: str) -> str:
        if self._hex_left:
            if ch not in "0123456789abcdefABCDEF":
                return "error"
            self._hex_left -= 1
            return "ok"
        if self._escape:
            if ch not in _ESCAPABLE:
                return "error"
            self._escape = False
            if ch == "u":
                self._hex_left = 4
            return "ok"
        if ch == "\\":
            self._escape = True
            return "ok"
        if ch == '"':
            self._state = "colon" if self._string_is_key else "after_value"
            return "ok"
        if ch < " ":
            return "error"
        return "ok"

    def _push_structural(self, ch: str) -> str:
        if ch in _JSON_WS:
            return "ok"
        if self._state in ("value", "array_first"):
            if self._state == "array_first" and ch == "]":
                return self._close("[")
            return self._start_value(ch)
        if self._state in ("key_first", "key_required"):
            if ch == '"':
                self._state = "in_string"
                self._string_is_key = True
                return "ok"
            if self._state == "key_first" and ch == "}":
                return self._close("{")
            return "error"
        if self._state == "colon":
            if ch == ":":
                self._state = "value"
                return "ok"
            return "error"
        if self._state == "after_value":
            if ch == ",":
                self._state = ("key_required"
                               if self._stack[-1] == "{" else "value")
                return "ok"
            if ch == "}":
                return self._close("{")
            if ch == "]":
                return self._close("[")
            return "error"
        return "error"

    def _start_value(self, ch: str) -> str:
        # A top-level value start is the earliest point the dialect is known.
        if ch == "{":
            self._confirm()
            self._stack.append("{")
            self._state = "key_first"
            return "ok"
        if ch == "[":
            self._confirm()
            self._stack.append("[")
            self._state = "array_first"
            return "ok"
        if not self._stack:
            return "error"  # Scalar arguments are not a RAW-mode payload.
        if ch == '"':
            self._confirm()
            self._state = "in_string"
            self._string_is_key = False
            return "ok"
        if ch in "-0123456789":
            self._confirm()
            self._state = "in_number"
            self._num_state = ("int_start" if ch == "-" else
                               "int_zero" if ch == "0" else "int")
            return "ok"
        if ch in _LITERALS:
            self._confirm()
            self._state = "in_literal"
            self._literal = _LITERALS[ch]
            self._literal_pos = 1
            return "ok"
        return "error"

    def _push_number(self, ch: str) -> str:
        if ch in _JSON_WS or ch in ",}]":
            if self._num_state in ("int_zero", "int", "frac", "exp"):
                self._state = "after_value"
                return self._push_structural(ch)
            return "error"
        num = self._num_state
        if num == "int_start":
            self._num_state = "int_zero" if ch == "0" else "int"
            return "ok" if ch in "0123456789" else "error"
        if num == "int_zero":
            if ch == ".":
                self._num_state = "frac_start"
            elif ch in "eE":
                self._num_state = "exp_start"
            else:
                return "error"
            return "ok"
        if num == "int":
            if ch == ".":
                self._num_state = "frac_start"
            elif ch in "eE":
                self._num_state = "exp_start"
            elif ch not in "0123456789":
                return "error"
            return "ok"
        if num in ("frac_start", "frac"):
            if ch in "0123456789":
                self._num_state = "frac"
            elif num == "frac" and ch in "eE":
                self._num_state = "exp_start"
            else:
                return "error"
            return "ok"
        if num == "exp_start" and ch in "+-":
            self._num_state = "exp_sign"
            return "ok"
        if ch in "0123456789":  # exp_start / exp_sign / exp
            self._num_state = "exp"
            return "ok"
        return "error"

    def _confirm(self) -> None:
        if not self.confirmed and len(self._stack) == 1:
            self.confirmed = True

    def _close(self, expected: str) -> str:
        if not self._stack or self._stack[-1] != expected:
            return "error"
        self._stack.pop()
        if not self._stack:
            self.done = True
            self.confirmed = True  # {} / [] close without any value start.
            return "done"
        self._state = "after_value"
        return "ok"


class _GenericToolParser:

    _BLOCK_RE = re.compile(
        "(" +
        "|".join(_family_block_regex(family)
                 for family in _MARKER_FAMILIES) + ")", re.S)

    def stream(self, tool_config: ToolConfig) -> "StreamingToolParser":
        """Create independent streaming state for one response."""
        return StreamingToolParser(self, tool_config)

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


class StreamingAssistantOutputParser:
    """Incrementally emit ordered reasoning, content, and tool-call events.

    Event-dict facade over ``StreamingToolParser`` for consumers that want
    complete tool calls (the Anthropic bridge): head/args/done triples are
    assembled into one "tool_call" event, while content and reasoning still
    stream incrementally.
    """

    def __init__(self, tool_config: ToolConfig, model_dir: str,
                 tool_parser: str, reasoning_parser: str) -> None:
        parser = _select_parser(model_dir, tool_parser)
        self._tools = parser.stream(
            tool_config) if tool_config.parse_output else None
        reasoning = REASONING_PARSERS.resolve(reasoning_parser, model_dir)
        self._reasoning = (reasoning.stream(
            allow_implicit=not tool_config.parse_output)
                           if reasoning else None)
        self._calls: Dict[int, ToolCall] = {}

    def feed(self, text: str) -> Iterable[Dict[str, Any]]:
        if self._tools is None:
            yield from self._split_reasoning(text)
            return
        yield from self._convert(self._tools.feed(text))

    def flush(self) -> Iterable[Dict[str, Any]]:
        if self._tools is not None:
            yield from self._convert(self._tools.flush())
        if self._reasoning:
            yield from self._reasoning_events(self._reasoning.flush())

    def _convert(
            self,
            events: Iterable["ToolStreamEvent"]) -> Iterable[Dict[str, Any]]:
        for event in events:
            if event.kind == "content":
                yield from self._split_reasoning(event.text)
            elif event.kind == "tool_head":
                self._calls[event.index] = ToolCall(id=event.call_id,
                                                    name=event.name,
                                                    arguments="")
            elif event.kind == "tool_args":
                self._calls[event.index].arguments += event.text
            elif event.kind == "tool_done":
                if self._reasoning:
                    yield from self._reasoning_events(self._reasoning.flush())
                yield {
                    "type": "tool_call",
                    "tool_call": self._calls.pop(event.index),
                }

    def _split_reasoning(self, text: str) -> Iterable[Dict[str, Any]]:
        if self._reasoning:
            yield from self._reasoning_events(self._reasoning.feed(text))
        elif text:
            yield {"type": "content", "text": text}

    @staticmethod
    def _reasoning_events(deltas: Iterable[Any]) -> Iterable[Dict[str, Any]]:
        for delta in deltas:
            yield {"type": delta.field, "text": delta.text}


def stream_assistant_output(
        tool_config: ToolConfig,
        model_dir: str,
        tool_parser: str = "auto",
        reasoning_parser: str = "none") -> StreamingAssistantOutputParser:
    """Create incremental parsing state for one assistant response."""
    return StreamingAssistantOutputParser(tool_config, model_dir, tool_parser,
                                          reasoning_parser)


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

    def names(self) -> List[str]:
        return ["auto"] + sorted(self._parsers)

    def get(self, model_dir: str, parser_name: str = "auto"):
        if parser_name == "auto":
            parser_name = _parser_name_for_model(model_dir)
        if parser_name not in self._parsers:
            available = ", ".join(self.names())
            raise KeyError(
                f"unknown tool parser {parser_name!r}; available: {available}")
        return self._parsers[parser_name]


_PARSERS = _ToolParserRegistry()


@dataclass(frozen=True)
class ToolStreamEvent:
    """One StreamingToolParser event; kind is "content", "tool_head",
    "tool_args", or "tool_done". text carries content or an arguments
    fragment."""

    kind: str
    text: str = ""
    index: int = 0
    call_id: str = ""
    name: str = ""


_GATE_CAP = 32
_BRACKET_OPENS = ("[TOOL_CALL]", "[TOOL_CALLS]")


def _gate_verdict(prefix: str) -> str:
    """Classify the opening bytes as "bare", "streaming", or "wait".

    Bare markerless payloads (JSON / pythonic / fenced) are only
    recognizable on complete text, so they take the buffered path.
    """
    stripped = prefix.lstrip()
    if not stripped:
        return "wait" if len(prefix) < _GATE_CAP else "streaming"
    first = stripped[0]
    if first == "{":
        return "bare"
    if first == "`":
        if stripped.startswith("```"):
            return "bare"
        return "wait" if "```".startswith(stripped) else "streaming"
    if first == "[":
        if any(stripped.startswith(token) for token in _BRACKET_OPENS):
            return "streaming"
        if any(token.startswith(stripped) for token in _BRACKET_OPENS):
            return "wait"
        return "bare"
    match = re.match(r"[A-Za-z_]\w*", stripped)
    if match:
        rest = stripped[match.end():].lstrip(" \t")
        if rest.startswith("("):
            return "bare"
        if not rest:
            return "wait" if len(stripped) < _GATE_CAP else "streaming"
    return "streaming"


_PROBE_CAP = 256
_MAX_CLOSE_HOLD = max(
    len(token) for family in _MARKER_FAMILIES for token in family.closes) - 1
# RAW mode accepts only the canonical Hermes/Qwen single-call shape: name
# first, then an *object* arguments value. Arrays go WHOLE -- the whole-text
# path normalizes an empty array to "{}", which streaming cannot foresee.
_RAW_HEAD_RE = re.compile(
    r'^\s*\{\s*"name"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"arguments"\s*:\s*(\{)',
    re.S)
_PARAM_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.S)


class StreamingToolParser:
    """Single-pass incremental splitter of model output into tool events.

    States: "gate" sniffs for bare markerless payloads ("bare" buffers them
    whole); "text" forwards content while holding back a possible partial
    marker; "probe" picks a call's mode before anything is emitted, so every
    degrade to "block" (buffer until close marker) is lossless; "raw" passes
    the arguments bytes through under _JsonScanner, then "raw_tail" swallows
    wrapper bytes ("raw_sink" after a mid-arguments violation); "param"
    emits a
    normalized arguments diff per closed qwen-xml parameter.

    flush() re-parses everything still held back with the whole-text parser,
    so EOF behavior matches the non-streaming path -- except calls whose head
    already streamed and cannot be recalled.
    """

    def __init__(self, whole_text_parser, tool_config: ToolConfig) -> None:
        self._whole = whole_text_parser
        self._config = tool_config
        self._buffer = ""
        self._state = "gate"
        self._next_index = 0
        # Per-call fields; _reset_call clears them between calls.
        self._family: Optional[_MarkerFamily] = None
        self._open_text = ""
        self._name = ""
        self._head_out = False
        self._scanner: Optional[_JsonScanner] = None
        self._args_prefix = ""
        self._raw_consumed = ""
        self._pending = ""
        self._param_types: Dict[str, str] = {}
        self._body = ""
        self._body_pos = 0
        self._acc: Dict[str, Any] = {}
        self._sent = ""
        self._poisoned = False

    def feed(self, text: str) -> Iterable[ToolStreamEvent]:
        self._buffer += text
        while True:
            if self._state == "gate":
                verdict = _gate_verdict(self._buffer)
                if verdict == "bare":
                    self._state = "bare"
                elif verdict == "streaming" or len(self._buffer) >= _GATE_CAP:
                    self._state = "text"
                    continue
                return
            if self._state == "bare":
                return
            if self._state == "text":
                found = self._find_open()
                if found is None:
                    hold = _longest_partial_token(self._buffer, _OPEN_TOKENS)
                    if len(self._buffer) > hold:
                        emit_len = len(self._buffer) - hold
                        fragment = self._buffer[:emit_len]
                        self._buffer = self._buffer[emit_len:]
                        yield ToolStreamEvent("content", fragment)
                    return
                start, token, family = found
                if start > 0:
                    yield ToolStreamEvent("content", self._buffer[:start])
                self._buffer = self._buffer[start + len(token):]
                self._family = family
                self._open_text = token
                self._state = "probe"
            elif self._state == "probe":
                if not (yield from self._probe()):
                    return
            elif self._state == "block":
                close = self._find_close(self._buffer)
                if close is None:
                    return
                end, close_token = close
                block = self._open_text + self._buffer[:end] + close_token
                self._buffer = self._buffer[end + len(close_token):]
                self._reset_call()
                yield from self._emit_block(block)
            elif self._state in ("raw_tail", "raw_sink"):
                close = self._find_close(self._buffer)
                if close is None:
                    self._buffer = self._buffer[-_MAX_CLOSE_HOLD:]
                    return
                end, close_token = close
                self._buffer = self._buffer[end + len(close_token):]
                yield self._done()
                self._reset_call()
            elif self._state == "raw":
                if not (yield from self._raw()):
                    return
            else:  # param
                if not (yield from self._param()):
                    return

    def _probe(self) -> Iterable[ToolStreamEvent]:
        """Pick the call's mode; nothing has been emitted for it yet, so
        every degrade to "block" is lossless."""
        if self._family.open_is_prefix:
            end = self._buffer.find(">")
            if end < 0:
                if len(self._buffer) > _PROBE_CAP:
                    self._state = "block"
                    return True
                return False
            if end == 0:
                # "<function=>" never matches the whole-text regex.
                yield ToolStreamEvent("content", self._open_text)
                self._reset_call()
                return True
            name = self._buffer[:end].strip()
            self._open_text += self._buffer[:end + 1]
            self._buffer = self._buffer[end + 1:]
            if _tool_name_allowed(name, self._config):
                self._start_param(name)
                yield self._head()
            else:
                self._state = "block"
            return True
        match = _RAW_HEAD_RE.match(self._buffer)
        if match:
            try:
                name = json.loads(f'"{match.group(1)}"')
            except ValueError:
                name = None
            if name and _tool_name_allowed(name, self._config):
                self._name = name
                args_start = match.end() - 1
                self._args_prefix = self._buffer[:args_start]
                self._buffer = self._buffer[args_start:]
                self._scanner = _JsonScanner()
                self._state = "raw"
            else:
                self._state = "block"
            return True
        if (self._find_close(self._buffer) is not None
                or len(self._buffer) > _PROBE_CAP):
            self._state = "block"
            return True
        return False

    def _raw(self) -> Iterable[ToolStreamEvent]:
        fragment = ""
        buffer = self._buffer
        for position, ch in enumerate(buffer):
            result = self._scanner.push(ch)
            if result == "error":
                if self._head_out:
                    logger.warning(
                        "streaming tool call %s: arguments stopped at a "
                        "strict-JSON violation; output may be truncated",
                        self._name)
                    self._buffer = buffer[position:]
                    self._state = "raw_sink"
                else:
                    # Lossless degrade: nothing emitted for this call yet.
                    self._buffer = (self._args_prefix + self._raw_consumed +
                                    buffer[position:])
                    self._state = "block"
                break
            self._raw_consumed += ch
            if self._head_out:
                fragment += ch
            else:
                self._pending += ch
                if self._scanner.confirmed:
                    yield self._head()
                    fragment = self._pending
                    self._pending = ""
            if result == "done":
                self._buffer = buffer[position + 1:]
                self._state = "raw_tail"
                break
        else:
            self._buffer = ""
        if fragment:
            yield self._args(fragment)
        return self._state != "raw"

    def _param(self) -> Iterable[ToolStreamEvent]:
        self._body += self._buffer
        self._buffer = ""
        close = self._find_close(self._body)
        limit = close[0] if close else len(self._body)
        fragment = ""
        for match in _PARAM_RE.finditer(self._body, self._body_pos, limit):
            self._body_pos = match.end()
            if self._poisoned:
                continue
            key = match.group(1).strip()
            self._acc[key] = _coerce_param(
                match.group(2).strip(), self._param_types.get(key))
            partial = _arguments_to_json(self._acc)[:-1]
            if not partial.startswith(self._sent):
                # A repeated key rewrote already-sent bytes; the diff is no
                # longer append-only, so arguments output must stop here.
                logger.warning(
                    "streaming tool call %s: non-monotonic parameter "
                    "update; arguments output stops here", self._name)
                self._poisoned = True
                continue
            fragment += partial[len(self._sent):]
            self._sent = partial
        if close is not None and not self._poisoned:
            fragment += _arguments_to_json(self._acc)[len(self._sent):]
        if fragment:
            yield self._args(fragment)
        if close is None:
            return False
        end, close_token = close
        yield self._done()
        self._buffer = self._body[end + len(close_token):]
        self._reset_call()
        return True

    def flush(self) -> Iterable[ToolStreamEvent]:
        if (self._state in ("raw", "raw_tail", "raw_sink", "param")
                and self._head_out):
            # The head already streamed and cannot be recalled.
            if self._state == "param" and not self._poisoned:
                tail = _arguments_to_json(self._acc)[len(self._sent):]
                if tail:
                    yield self._args(tail)
            yield self._done()
            withheld = ""
        elif self._state == "raw":
            # Truncated before anything streamed for this call: rejoin the
            # whole-text re-parse with every withheld byte.
            withheld = (self._open_text + self._args_prefix +
                        self._raw_consumed + self._buffer)
        elif self._state in ("probe", "block"):
            withheld = self._open_text + self._buffer
        else:
            withheld = self._buffer
        self._reset_call()
        self._buffer = ""
        if not withheld:
            return
        events, _ = self._whole.parse(withheld, self._config)
        for event in events:
            if event["type"] == "content":
                if event["text"]:
                    yield ToolStreamEvent("content", event["text"])
            else:
                yield from self._emit_call(event["tool_call"])

    def _head(self) -> ToolStreamEvent:
        self._head_out = True
        index = self._next_index
        self._next_index += 1
        return ToolStreamEvent("tool_head",
                               index=index,
                               call_id=_new_call_id(),
                               name=self._name)

    def _args(self, text: str) -> ToolStreamEvent:
        return ToolStreamEvent("tool_args", text, index=self._next_index - 1)

    def _done(self) -> ToolStreamEvent:
        return ToolStreamEvent("tool_done", index=self._next_index - 1)

    def _start_param(self, name: str) -> None:
        self._name = name
        self._param_types = _param_types_for(name, self._config)
        self._body = ""
        self._body_pos = 0
        self._acc = {}
        self._sent = ""
        self._poisoned = False
        self._state = "param"

    def _reset_call(self) -> None:
        self._state = "text"
        self._family = None
        self._open_text = ""
        self._name = ""
        self._head_out = False
        self._scanner = None
        self._args_prefix = ""
        self._raw_consumed = ""
        self._pending = ""

    def _find_open(self):
        best = None
        for family in _MARKER_FAMILIES:
            for token in family.opens:
                idx = self._buffer.find(token)
                if idx >= 0 and (best is None or idx < best[0] or
                                 (idx == best[0]
                                  and len(token) > len(best[1]))):
                    best = (idx, token, family)
        return best

    def _find_close(self, text: str):
        best = None
        for token in self._family.closes:
            idx = text.find(token)
            if idx >= 0 and (best is None or idx < best[0]):
                best = (idx, token)
        return best

    def _emit_block(self, block: str) -> Iterable[ToolStreamEvent]:
        calls = _parse_tool_block(block, self._config)
        if not calls:
            # Malformed: byte-identical with the whole-text malformed branch.
            yield ToolStreamEvent("content", block)
            return
        for call in calls:
            yield from self._emit_call(call)

    def _emit_call(self, call: ToolCall) -> Iterable[ToolStreamEvent]:
        index = self._next_index
        self._next_index += 1
        yield ToolStreamEvent("tool_head",
                              index=index,
                              call_id=call.id,
                              name=call.name)
        if call.arguments:
            yield ToolStreamEvent("tool_args", call.arguments, index=index)
        yield ToolStreamEvent("tool_done", index=index)


def _select_parser(model_dir: str, parser_name: str = "auto"):
    return _PARSERS.get(model_dir, parser_name)


def list_tool_parsers() -> List[str]:
    return _PARSERS.names()


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
    for pattern in _STRIP_TAG_PATTERNS:
        match = pattern.match(body)
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


def _param_types_for(name: str, tool_config: ToolConfig) -> Dict[str, str]:
    """Map each parameter name -> its JSON-schema ``type`` for the named function."""
    for tool in tool_config.tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(fn, dict) and fn.get("name") == name:
            props = (fn.get("parameters") or {}).get("properties")
            if isinstance(props, dict):
                return {
                    k: v.get("type")
                    for k, v in props.items()
                    if isinstance(v, dict) and isinstance(v.get("type"), str)
                }
    return {}


def _coerce_param(value: str, ptype: Optional[str]) -> Any:
    """Coerce an XML-extracted string to its declared JSON-schema type.

    Qwen's ``<parameter=...>...</parameter>`` values are always text; without
    this the arguments stay stringly-typed (``base="10"``) and fail strict
    consumers such as BFCL AST matching, which expects ``base=10``.
    """
    v = value.strip()
    try:
        if ptype == "integer":
            return int(v)
        if ptype == "number":
            f = float(v)
            return int(f) if f.is_integer() else f
        if ptype == "boolean":
            return v.lower() in ("true", "1", "yes")
        if ptype in ("array", "object"):
            return json.loads(v)
    except (ValueError, json.JSONDecodeError):
        return value
    return value


def _parse_qwen_xml_calls(text: str,
                          tool_config: ToolConfig) -> List[ToolCall]:
    calls = []
    for match in re.finditer(r"<function=([^>]+)>(.*?)</function>",
                             text,
                             flags=re.S):
        name = match.group(1).strip()
        if not _tool_name_allowed(name, tool_config):
            continue
        ptypes = _param_types_for(name, tool_config)
        args: Dict[str, Any] = {}
        for param in _PARAM_RE.finditer(match.group(2)):
            pname = param.group(1).strip()
            args[pname] = _coerce_param(
                param.group(2).strip(), ptypes.get(pname))
        calls.append(
            ToolCall(id=_new_call_id(),
                     name=name,
                     arguments=_arguments_to_json(args)))
    return calls


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
