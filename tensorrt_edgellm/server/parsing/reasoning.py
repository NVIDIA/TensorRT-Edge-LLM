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
"""Reasoning parser registry and incremental delimiter parser."""

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ReasoningDelta:
    """One incremental reasoning or visible-content fragment."""

    field: str
    text: str


class ReasoningParser:
    """Parser for models that delimit reasoning with thinking tokens."""

    def __init__(self,
                 start_token: str = "<think>",
                 end_token: str = "</think>",
                 *,
                 always_enabled: bool = False) -> None:
        self.start_token = start_token
        self.end_token = end_token
        self.always_enabled = always_enabled

    def extract(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Split output even when generation starts inside a reasoning span."""
        _, start, after_start = text.partition(self.start_token)
        candidate = after_start if start else text
        if self.end_token not in candidate:
            return (candidate or None), None
        reasoning, _, content = candidate.partition(self.end_token)
        return (reasoning or None), (content or None)

    def stream(self,
               *,
               allow_implicit: bool = True) -> "StreamingReasoningParser":
        """Create independent state for one streaming response."""
        return StreamingReasoningParser(self.start_token,
                                        self.end_token,
                                        allow_implicit=allow_implicit)


class StreamingReasoningParser:
    """Incrementally split reasoning when delimiters cross chunk boundaries."""

    def __init__(self,
                 start_token: str,
                 end_token: str,
                 *,
                 allow_implicit: bool = True) -> None:
        self._start = start_token
        self._end = end_token
        self._reasoning = allow_implicit
        self._allow_implicit = allow_implicit
        self._at_start = True
        self._buffer = ""

    def feed(self, text: str) -> Iterable[ReasoningDelta]:
        """Consume a generated chunk and yield every complete fragment."""
        self._buffer += text
        if self._at_start:
            if self._buffer.startswith(self._start):
                self._buffer = self._buffer[len(self._start):]
                self._reasoning = True
                self._at_start = False
            elif len(self._buffer) >= len(self._start):
                self._reasoning = self._allow_implicit
                self._at_start = False

        while self._buffer:
            marker = self._end if self._reasoning else self._start
            index = self._buffer.find(marker)
            if index >= 0:
                if index:
                    yield ReasoningDelta(
                        "reasoning" if self._reasoning else "content",
                        self._buffer[:index],
                    )
                self._buffer = self._buffer[index + len(marker):]
                self._reasoning = not self._reasoning
                continue

            keep = max(0, len(marker) - 1)
            if len(self._buffer) <= keep:
                break
            emit = self._buffer[:-keep] if keep else self._buffer
            self._buffer = self._buffer[-keep:] if keep else ""
            if emit:
                yield ReasoningDelta(
                    "reasoning" if self._reasoning else "content", emit)

    def flush(self) -> Iterable[ReasoningDelta]:
        """Yield buffered text after generation finishes."""
        if self._buffer:
            if self._at_start:
                self._reasoning = self._allow_implicit
                self._at_start = False
            yield ReasoningDelta("reasoning" if self._reasoning else "content",
                                 self._buffer)
            self._buffer = ""


_MUSE_REASONING_START = " to=self<|message|>"
_MUSE_CHANNEL_START = "<|eom|><|start|>assistant to="
_MUSE_MESSAGE_START = "<|message|>"
_MUSE_END = "<|eot|>"


class MuseGlimmerReasoningParser(ReasoningParser):
    """Split Muse-Glimmer's ATEM assistant channels."""

    def __init__(self) -> None:
        super().__init__(_MUSE_REASONING_START,
                         _MUSE_CHANNEL_START,
                         always_enabled=True)

    @staticmethod
    def _drop_recipient(text: str) -> str:
        if _MUSE_MESSAGE_START in text:
            text = text.partition(_MUSE_MESSAGE_START)[2]
        return text.removesuffix(_MUSE_END)

    def extract(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        if text.startswith(self.start_token):
            candidate = text[len(self.start_token):]
        elif self.start_token in text:
            candidate = text.partition(self.start_token)[2]
        else:
            content = self._drop_recipient(text)
            return None, content or None
        if self.end_token not in candidate:
            return candidate or None, None
        reasoning, _, content = candidate.partition(self.end_token)
        content = self._drop_recipient(content)
        return reasoning or None, content or None

    def stream(
            self,
            *,
            allow_implicit: bool = True
    ) -> "MuseGlimmerStreamingReasoningParser":
        del allow_implicit
        return MuseGlimmerStreamingReasoningParser()


class MuseGlimmerStreamingReasoningParser:
    """Incrementally split ATEM reasoning and recipient headers."""

    def __init__(self) -> None:
        self._state = "header"
        self._buffer = ""

    def feed(self, text: str) -> Iterable[ReasoningDelta]:
        self._buffer += text
        while self._buffer:
            if self._state == "header":
                if _MUSE_REASONING_START.startswith(self._buffer):
                    return
                if self._buffer.startswith(_MUSE_REASONING_START):
                    self._buffer = self._buffer[len(_MUSE_REASONING_START):]
                    self._state = "reasoning"
                    continue
                if self._buffer.startswith(" to="):
                    marker = self._buffer.find(_MUSE_MESSAGE_START)
                    if marker < 0:
                        return
                    self._buffer = self._buffer[marker +
                                                len(_MUSE_MESSAGE_START):]
                self._state = "content"
                continue

            if self._state == "reasoning":
                marker = self._buffer.find(_MUSE_CHANNEL_START)
                if marker >= 0:
                    if marker:
                        yield ReasoningDelta("reasoning",
                                             self._buffer[:marker])
                    self._buffer = self._buffer[marker +
                                                len(_MUSE_CHANNEL_START):]
                    self._state = "recipient"
                    continue
                keep = len(_MUSE_CHANNEL_START) - 1
                if len(self._buffer) <= keep:
                    return
                yield ReasoningDelta("reasoning", self._buffer[:-keep])
                self._buffer = self._buffer[-keep:]
                return

            if self._state == "recipient":
                marker = self._buffer.find(_MUSE_MESSAGE_START)
                if marker < 0:
                    return
                self._buffer = self._buffer[marker + len(_MUSE_MESSAGE_START):]
                self._state = "content"
                continue

            marker = self._buffer.find(_MUSE_END)
            if marker >= 0:
                if marker:
                    yield ReasoningDelta("content", self._buffer[:marker])
                self._buffer = self._buffer[marker + len(_MUSE_END):]
                continue
            keep = len(_MUSE_END) - 1
            if len(self._buffer) <= keep:
                return
            yield ReasoningDelta("content", self._buffer[:-keep])
            self._buffer = self._buffer[-keep:]
            return

    def flush(self) -> Iterable[ReasoningDelta]:
        if not self._buffer:
            return
        field = "reasoning" if self._state == "reasoning" else "content"
        if self._state in {"header", "recipient"}:
            self._buffer = MuseGlimmerReasoningParser._drop_recipient(
                self._buffer)
        elif self._state == "content":
            self._buffer = self._buffer.removesuffix(_MUSE_END)
        if self._buffer:
            yield ReasoningDelta(field, self._buffer)
        self._buffer = ""


class ReasoningParserRegistry:
    """Resolve explicit parser names or infer one from model metadata.

    Automatic entries are limited to model families whose provider chat
    templates emit ``<think>...</think>`` spans. Explicit parser selection is
    still available for derived checkpoints with a custom template.
    """

    def __init__(self) -> None:
        thinking = ReasoningParser()
        self._parsers: Dict[str, Optional[ReasoningParser]] = {
            "none": None,
            "qwen3": thinking,
            "deepseek_r1": thinking,
            "muse_glimmer": MuseGlimmerReasoningParser(),
            "nemotron": thinking,
        }

    def names(self) -> List[str]:
        """Return accepted command-line parser names."""
        return ["auto"] + sorted(self._parsers)

    def resolve(self, name: str, model_dir: str) -> Optional[ReasoningParser]:
        """Resolve a configured parser for a local checkpoint directory."""
        if name == "auto":
            name = self._for_model(model_dir)
        if name not in self._parsers:
            available = ", ".join(self.names())
            raise KeyError(
                f"unknown reasoning parser {name!r}; available: {available}")
        return self._parsers[name]

    @staticmethod
    def _for_model(model_dir: str) -> str:
        """Infer a parser from config metadata, with basename as fallback."""
        identifiers = _model_identifiers(model_dir)
        if any("muse_glimmer" in value for value in identifiers):
            return "muse_glimmer"
        if any("nemotron" in value for value in identifiers):
            return "nemotron"
        if any("deepseek" in value for value in identifiers):
            return "deepseek_r1"
        if any("qwen3" in value for value in identifiers):
            return "qwen3"
        return "none"


def _model_identifiers(model_dir: str) -> List[str]:
    """Read architecture identifiers without relying on checkpoint location."""
    identifiers = [os.path.basename(os.path.normpath(model_dir)).lower()]
    config_path = os.path.join(model_dir, "config.json")
    try:
        with open(config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, TypeError, ValueError):
        return identifiers

    def collect(value) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "model_type" and isinstance(item, str):
                    identifiers.append(item.lower())
                elif key == "architectures" and isinstance(item, list):
                    identifiers.extend(name.lower() for name in item
                                       if isinstance(name, str))
                elif isinstance(item, (dict, list)):
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(config)
    return identifiers


REASONING_PARSERS = ReasoningParserRegistry()
