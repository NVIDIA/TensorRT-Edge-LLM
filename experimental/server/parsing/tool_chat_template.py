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
"""Tool-aware chat template formatting for agentic workloads."""

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

CONTEXT_REUSE_PROBE = "context reuse probe"


class ToolChatTemplateError(ValueError):
    """Tool chat template formatting failed."""


def needs_tool_chat_template(
    messages: Sequence[Dict[str, Any]],
    tools: Optional[Sequence[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
) -> bool:
    """Return whether a request needs full chat template formatting."""

    if tools:
        return True
    if tool_choice and tool_choice != "none":
        return True

    for msg in messages:
        if msg.get("role") == "tool":
            return True
        if msg.get("tool_calls") or msg.get("function_call"):
            return True
    return False


def _json_loads_if_possible(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _normalize_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize OpenAI tool calls for HF chat templates."""
    normalized = copy.deepcopy(tool_call)
    function = normalized.get("function")
    if isinstance(function, dict):
        function["arguments"] = _json_loads_if_possible(
            function.get("arguments", {}))
        return normalized

    if "arguments" in normalized:
        normalized["arguments"] = _json_loads_if_possible(
            normalized["arguments"])
    return normalized


def _flatten_content_blocks(content: Any) -> Any:
    """Collapse a non-empty array of typed text blocks into the plain string
    HF chat templates expect. Anything else -- empty lists, media blocks, raw
    strings, JSON tool-result lists -- passes through unchanged so the later
    role-specific handling (json.dumps of tool results, video normalization)
    still applies."""
    if not isinstance(content, list) or not content:
        return content
    parts: List[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif (isinstance(item, dict)
              and item.get("type") in ("text", "input_text")
              and isinstance(item.get("text"), str)):
            parts.append(item["text"])
        else:
            return content
    return "\n".join(p for p in parts if p)


def normalize_messages_for_tools(
        messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a template-friendly copy of OpenAI-style messages."""
    normalized_messages: List[Dict[str, Any]] = []

    for raw_msg in messages:
        msg = copy.deepcopy(raw_msg)
        # Agentic clients (Claude Code, OpenClaw) send content as
        # [{"type":"text",...}] arrays; collapse pure-text arrays to a string
        # (media/JSON-list content passes through) or the template renders an
        # empty turn and the model never sees the message.
        if isinstance(msg.get("content"), list):
            msg["content"] = _flatten_content_blocks(msg["content"])

        # Normalize OpenAI URL blocks to the HF processor content contract.
        if isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if not isinstance(item, dict):
                    continue
                content_type = item.get("type")
                if content_type not in ("image_url", "video_url", "audio_url"):
                    if content_type == "input_audio":
                        payload = item.pop("input_audio", {})
                        item["type"] = "audio"
                        item["audio"] = payload.get("data", "")
                    continue
                ref = item.pop(content_type, None)
                media_type = content_type.removesuffix("_url")
                item["type"] = media_type
                item[media_type] = (ref.get("url", "") if isinstance(
                    ref, dict) else ref)

        if msg.get("function_call") and not msg.get("tool_calls"):
            function_call = msg.pop("function_call")
            if isinstance(function_call, dict):
                msg["tool_calls"] = [{
                    "type": "function",
                    "function": {
                        "name":
                        function_call.get("name", ""),
                        "arguments":
                        _json_loads_if_possible(
                            function_call.get("arguments", {})),
                    },
                }]

        if msg.get("tool_calls"):
            msg["tool_calls"] = [
                _normalize_tool_call(tc) for tc in msg["tool_calls"]
                if isinstance(tc, dict)
            ]

        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if content is None:
                msg["content"] = ""
            elif not isinstance(content, str):
                msg["content"] = json.dumps(content, ensure_ascii=False)

        normalized_messages.append(msg)

    return normalized_messages


class ToolChatTemplateFormatter:
    """Apply HF chat templates for tool-aware requests."""

    def __init__(
        self,
        template_dirs: Iterable[str],
        *,
        template_owner: Optional[Any] = None,
    ) -> None:
        self._template_dirs = [
            str(Path(d)) for d in template_dirs if d and Path(d).is_dir()
        ]
        self._template_owner = template_owner
        self._replay_tail_lengths: Dict[Tuple[Optional[bool], bool,
                                              Optional[str]], int] = {}

    def _load_template_owner(self) -> Any:
        if self._template_owner is not None:
            return self._template_owner

        try:
            from transformers import (AutoProcessor, AutoTokenizer,
                                      PreTrainedTokenizerFast)
        except ImportError as exc:
            raise ToolChatTemplateError(
                "transformers is required for tool-aware chat templates; "
                "install tensorrt-edgellm[server-tools].") from exc

        # Some checkpoints expose a fast tokenizer without an AutoProcessor.
        loaders = (AutoProcessor, AutoTokenizer, PreTrainedTokenizerFast)

        errors = []
        for template_dir in self._template_dirs:
            for loader in loaders:
                try:
                    owner = loader.from_pretrained(template_dir,
                                                   trust_remote_code=False)
                except Exception as exc:
                    errors.append(f"{loader.__name__}({template_dir}): {exc}")
                    continue
                if hasattr(owner, "apply_chat_template"):
                    self._template_owner = owner
                    return owner

        detail = "; ".join(errors[-3:]) if errors else "no template dirs"
        raise ToolChatTemplateError(
            "Could not load a tokenizer or processor with apply_chat_template "
            f"from {self._template_dirs}: {detail}")

    def format(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: bool = True,
        add_generation_prompt: bool = True,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        """Format messages and tools into a model-native prompt."""
        owner = self._load_template_owner()
        normalized_messages = normalize_messages_for_tools(messages)

        kwargs: Dict[str, Any] = {
            "tools": list(tools or []),
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        kwargs["parallel_tool_calls"] = parallel_tool_calls

        try:
            prompt = owner.apply_chat_template(normalized_messages, **kwargs)
        except TypeError as exc:
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("enable_thinking", None)
            if parallel_tool_calls:
                retry_kwargs.pop("parallel_tool_calls", None)
            if retry_kwargs == kwargs:
                raise ToolChatTemplateError(
                    f"Failed to apply tool-aware chat template: {exc}"
                ) from exc
            try:
                prompt = owner.apply_chat_template(normalized_messages,
                                                   **retry_kwargs)
            except Exception as retry_exc:
                raise ToolChatTemplateError(
                    "Failed to apply tool-aware chat template: "
                    f"{retry_exc}") from retry_exc
        except Exception as exc:
            raise ToolChatTemplateError(
                f"Failed to apply tool-aware chat template: {exc}") from exc

        if not isinstance(prompt, str):
            raise ToolChatTemplateError(
                "Tool-aware chat template returned non-string prompt. "
                "Use tokenize=False-compatible tokenizer/processor templates.")
        return prompt

    def count_tokens(self, text: str) -> Optional[int]:
        """Count tokens of a rendered prompt (no special-token wrapper — the
        rendered text already carries them). None if encoding unavailable."""
        owner = self._load_template_owner()
        tokenizer = getattr(owner, "tokenizer", owner)
        encode = getattr(tokenizer, "encode", None)
        if not callable(encode):
            return None
        try:
            return len(encode(text, add_special_tokens=False))
        except TypeError:
            return len(encode(text))

    def _encode_prompt(self, prompt: str) -> List[int]:
        owner = self._load_template_owner()
        tokenizer = getattr(owner, "tokenizer", owner)
        if not hasattr(tokenizer, "encode"):
            raise ToolChatTemplateError(
                "Tool-aware chat template owner has no tokenizer encode method."
            )

        try:
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        except Exception as exc:
            raise ToolChatTemplateError(
                f"Failed to tokenize tool-aware prompt: {exc}") from exc
        if not isinstance(token_ids, list) or not all(
                isinstance(token_id, int) for token_id in token_ids):
            raise ToolChatTemplateError(
                "Tool-aware prompt tokenizer returned invalid token IDs.")
        return token_ids

    def format_with_replay_tail(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: bool = True,
        enable_thinking: Optional[bool] = None,
    ) -> Tuple[str, int]:
        """Format a prompt and find its unstable multi-turn token tail.

        The tail is the suffix of the generation prompt that the next turn's
        render of the same history does not reproduce, measured as the longest
        common prefix against a probe render of the committed conversation.
        It is non-empty exactly when the template folds a thinking marker into
        the generation prompt but drops it from history, e.g. 4 tokens for
        Qwen3 ``<think>\\n\\n</think>\\n\\n`` under ``enable_thinking=False``
        and 2 tokens for Qwen3-Next-Thinking's ``<think>\\n``.

        The result is memoized on ``(enable_thinking, bool(tools), last_role)``
        rather than on the tool schemas themselves. That is safe for every
        template in the supported family because the generation-prompt suffix
        is a fixed string: tool definitions are rendered into the system turn,
        which sits inside the common prefix and therefore cancels out of the
        difference. A template that folded tool-specific text into the
        generation prompt would break this assumption and needs the schema in
        the key.
        """
        prompt = self.format(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        cache_key = (enable_thinking, bool(tools), messages[-1].get("role"))
        if cache_key in self._replay_tail_lengths:
            return prompt, self._replay_tail_lengths[cache_key]

        probe_messages = list(messages) + [{
            "role": "assistant",
            "reasoning_content": CONTEXT_REUSE_PROBE,
            "content": "",
        }]
        committed_prompt = self.format(
            probe_messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        prompt_token_ids = self._encode_prompt(prompt)
        committed_prompt_token_ids = self._encode_prompt(committed_prompt)
        common_prefix_length = 0
        for prompt_token_id, committed_token_id in zip(
                prompt_token_ids, committed_prompt_token_ids):
            if prompt_token_id != committed_token_id:
                break
            common_prefix_length += 1
        if common_prefix_length == 0:
            raise ToolChatTemplateError(
                "Tool-aware prompt has no stable token prefix for context reuse."
            )

        # MTP needs one token after the captured predecessor as a stable
        # successor proof, so replay starts one token before the LCP boundary.
        replay_tail_length = (len(prompt_token_ids) - common_prefix_length + 1)
        self._replay_tail_lengths[cache_key] = replay_tail_length
        return prompt, replay_tail_length
