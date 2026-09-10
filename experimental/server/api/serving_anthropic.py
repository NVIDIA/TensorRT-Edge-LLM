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
"""Anthropic Messages serving over the shared Edge-LLM engine client."""

import asyncio
import contextlib
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Iterable, Optional

from ..runtime.engine_client import EngineClient, PreparedRequest
from . import anthropic_compat as protocol
from .errors import ServerError
from .protocol import ChatCompletionRequest
from .serving_chat import IM_END_TOKEN, OpenAIServingChat, PreparedChatRequest


@dataclass(frozen=True)
class PreparedAnthropicStream:
    request: ChatCompletionRequest
    chat: PreparedChatRequest
    engine: PreparedRequest
    message_id: str


class _ContentBlockStream:
    """Serialize ordered parser events as Anthropic content blocks."""

    def __init__(self) -> None:
        self._index = 0
        self._open_type: Optional[str] = None
        self.tool_calls = 0
        self.has_blocks = False

    def feed(self, events: Iterable[Dict[str, Any]]) -> Iterable[str]:
        for parsed in events:
            event_type = parsed["type"]
            if event_type == "tool_call":
                yield from self._close_block()
                call = parsed["tool_call"]
                yield protocol.event(
                    "content_block_start", {
                        "index": self._index,
                        "content_block": {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": {},
                        },
                    })
                yield protocol.event(
                    "content_block_delta", {
                        "index": self._index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": call.arguments,
                        },
                    })
                yield protocol.event("content_block_stop",
                                     {"index": self._index})
                self._index += 1
                self.tool_calls += 1
                self.has_blocks = True
                continue

            text = parsed.get("text", "")
            if not text:
                continue
            block_type = "thinking" if event_type == "reasoning" else "text"
            if self._open_type != block_type:
                yield from self._close_block()
                content_block = ({
                    "type": "thinking",
                    "thinking": "",
                    "signature": "",
                } if block_type == "thinking" else {
                    "type": "text",
                    "text": "",
                })
                yield protocol.event("content_block_start", {
                    "index": self._index,
                    "content_block": content_block,
                })
                self._open_type = block_type
                self.has_blocks = True
            delta = ({
                "type": "thinking_delta",
                "thinking": text,
            } if block_type == "thinking" else {
                "type": "text_delta",
                "text": text,
            })
            yield protocol.event("content_block_delta", {
                "index": self._index,
                "delta": delta,
            })

    def finish(self) -> Iterable[str]:
        if not self.has_blocks:
            yield protocol.event(
                "content_block_start", {
                    "index": self._index,
                    "content_block": {
                        "type": "text",
                        "text": ""
                    },
                })
            self._open_type = "text"
        yield from self._close_block()

    def _close_block(self) -> Iterable[str]:
        if self._open_type is None:
            return
        yield protocol.event("content_block_stop", {"index": self._index})
        self._index += 1
        self._open_type = None


class AnthropicServingMessages:
    """Translate Anthropic protocol data without duplicating inference."""

    def __init__(self, client: EngineClient, chat: OpenAIServingChat) -> None:
        self._client = client
        self._chat = chat

    def _request(self,
                 body: Dict[str, Any],
                 *,
                 require_max_tokens: bool = True) -> ChatCompletionRequest:
        messages, tools, tool_choice, sampling = protocol.convert_request(
            body, require_max_tokens=require_max_tokens)
        raw_choice = body.get("tool_choice") or {}
        return ChatCompletionRequest(
            model=body.get("model"),
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=not bool(
                raw_choice.get("disable_parallel_tool_use", False)),
            stream=False,
            **sampling,
        )

    async def create_message(self, body: Dict[str, Any]) -> Dict[str, Any]:
        request = self._request(body)
        response = await self._chat.create_chat_completion(request)
        choice = response.choices[0]
        message = choice.message
        return {
            "id":
            f"msg_{uuid.uuid4().hex}",
            "type":
            "message",
            "role":
            "assistant",
            "model":
            response.model,
            "content":
            protocol.build_content_blocks(
                message.content,
                message.tool_calls or [],
                message.reasoning_content,
            ),
            "stop_reason":
            protocol.convert_stop_reason(choice.finish_reason),
            "stop_sequence":
            None,
            "usage":
            protocol.usage(response.usage.prompt_tokens,
                           response.usage.completion_tokens),
        }

    async def count_tokens(self, body: Dict[str, Any]) -> int:
        request = self._request(body, require_max_tokens=False)
        chat = self._chat.prepare_request(request)
        count = await self._client.count_prompt_tokens(
            request.messages,
            tool_config=chat.tool_config,
            enable_thinking=request.enable_thinking,
        )
        return count or 0

    async def prepare_stream(self, body: Dict[str,
                                              Any]) -> PreparedAnthropicStream:
        request = self._request(body)
        chat = self._chat.prepare_request(request)
        engine = await self._chat.prepare_engine_request(request, chat)
        return PreparedAnthropicStream(
            request=request,
            chat=chat,
            engine=engine,
            message_id=f"msg_{uuid.uuid4().hex}",
        )

    async def stream_message(
        self,
        prepared: PreparedAnthropicStream,
    ) -> AsyncGenerator[str, None]:
        parser = self._chat.stream_output_parser(prepared.chat)
        blocks = _ContentBlockStream()
        prompt_tokens = None
        completion_tokens = 0
        finish_reason = "stop"
        started = False
        native_stream = self._client.stream(
            prepared.request.messages,
            prepared.chat.sampling,
            tools=prepared.chat.tool_config.tools,
            tool_choice=prepared.chat.tool_config.tool_choice,
            prepared=prepared.engine,
        )
        next_delta = None
        try:
            while True:
                if next_delta is None:
                    next_delta = asyncio.create_task(anext(native_stream))
                done, _ = await asyncio.wait({next_delta}, timeout=5.0)
                if not done:
                    if started:
                        yield protocol.event("ping", {})
                    continue
                try:
                    delta = next_delta.result()
                except StopAsyncIteration:
                    next_delta = None
                    break
                next_delta = None
                completion_tokens += len(delta.token_ids)
                if delta.prompt_tokens is not None:
                    prompt_tokens = delta.prompt_tokens
                if delta.finished:
                    finish_reason = delta.finish_reason or "stop"
                if not started:
                    for chunk in protocol.message_start_events(
                            prepared.message_id, self._client.model_name,
                            prompt_tokens):
                        yield chunk
                    started = True
                    if delta.text:
                        text = delta.text.replace(IM_END_TOKEN, "")
                        for chunk in blocks.feed(parser.feed(text)):
                            yield chunk
                elif delta.text:
                    text = delta.text.replace(IM_END_TOKEN, "")
                    for chunk in blocks.feed(parser.feed(text)):
                        yield chunk
        except asyncio.CancelledError:
            raise
        except ServerError as exc:
            status = 529 if exc.status_code == 429 else exc.status_code
            yield protocol.event("error",
                                 protocol.error_payload(status, str(exc)))
            return
        except Exception as exc:
            yield protocol.event("error",
                                 protocol.error_payload(500, str(exc)))
            return
        finally:
            if next_delta is not None:
                next_delta.cancel()
                with contextlib.suppress(BaseException):
                    await next_delta
            with contextlib.suppress(BaseException):
                await native_stream.aclose()

        if not started:
            for chunk in protocol.message_start_events(prepared.message_id,
                                                       self._client.model_name,
                                                       prompt_tokens):
                yield chunk
        for chunk in blocks.feed(parser.flush()):
            yield chunk
        for chunk in blocks.finish():
            yield chunk
        if blocks.tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"
        yield protocol.event(
            "message_delta", {
                "delta": {
                    "stop_reason": protocol.convert_stop_reason(finish_reason),
                    "stop_sequence": None,
                },
                "usage": {
                    "output_tokens": completion_tokens
                },
            })
        yield protocol.event("message_stop", {})
