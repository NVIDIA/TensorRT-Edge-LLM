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
"""OpenAI chat-completion serving logic independent of FastAPI routing."""

import asyncio
import base64
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional

from ..config import ApiConfig
from ..parsing.reasoning import REASONING_PARSERS
from ..parsing.tool_calling import (StreamingAssistantOutputParser, ToolConfig,
                                    _select_parser, list_tool_parsers,
                                    parse_assistant_output,
                                    stream_assistant_output,
                                    validate_tool_request)
from ..runtime.engine import (OMNI_AUDIO_SAMPLE_RATE, AudioParams,
                              SamplingParams)
from ..runtime.engine_client import EngineClient, PreparedRequest
from .errors import (InvalidRequestError, ModelNotFoundError, ServerError,
                     UnsupportedFeatureError)
from .protocol import (ChatCompletionChoice, ChatCompletionMessage,
                       ChatCompletionRequest, ChatCompletionResponse,
                       ChatCompletionStreamChoice,
                       ChatCompletionStreamResponse, DeltaMessage, UsageInfo)

logger = logging.getLogger("edgellm.server.chat")

IM_END_TOKEN = "<|im_end|>"


@dataclass(frozen=True)
class PreparedChatRequest:
    sampling: SamplingParams
    tool_config: ToolConfig
    reasoning_parser: str
    audio_params: Optional[AudioParams] = None


def _entry_to_openai(entry) -> Dict[str, Any]:
    token = getattr(entry, "token", "")
    raw_bytes = list(getattr(entry, "bytes", []))
    return {
        "token": token,
        "token_id": entry.token_id,
        "bytes": raw_bytes,
        "logprob": float(entry.logprob),
    }


def _format_logprob_steps(token_ids, steps,
                          include_top: bool) -> Optional[Dict[str, Any]]:
    if not steps:
        return None
    content = []
    for token_id, step in zip(token_ids, steps):
        top = [_entry_to_openai(entry) for entry in step]
        chosen = next(
            (candidate
             for candidate in top if candidate["token_id"] == token_id), None)
        content.append({
            "token": chosen["token"] if chosen else "",
            "token_id": token_id,
            "bytes": chosen["bytes"] if chosen else [],
            "logprob": chosen["logprob"] if chosen else None,
            "top_logprobs": top if include_top else [],
        })
    return {"content": content}


def _usage(prompt_tokens: Optional[int], completion_tokens: int) -> UsageInfo:
    prompt_count = prompt_tokens or 0
    return UsageInfo(
        prompt_tokens=prompt_count,
        completion_tokens=completion_tokens,
        total_tokens=prompt_count + completion_tokens,
    )


def _sse(model) -> str:
    return "data: " + model.model_dump_json(exclude_none=False) + "\n\n"


def _sse_error(error: ServerError) -> str:
    return "data: " + json.dumps(error.payload()) + "\n\n"


class OpenAIServingChat:
    """Validate OpenAI requests and translate them to ``EngineClient`` calls."""

    def __init__(self, engine_client: EngineClient, config: ApiConfig) -> None:
        self._client = engine_client
        self._config = config
        self._model_dir = engine_client.llm.model_dir
        # Fail during startup instead of after the first request.
        REASONING_PARSERS.resolve(config.reasoning_parser, self._model_dir)
        if config.tool_call_parser not in list_tool_parsers():
            available = ", ".join(list_tool_parsers())
            raise ValueError(
                f"unknown tool parser {config.tool_call_parser!r}; "
                f"available: {available}")

    def prepare_request(self,
                        request: ChatCompletionRequest) -> PreparedChatRequest:
        capabilities = self._client.capabilities
        if not capabilities.chat:
            raise UnsupportedFeatureError(
                "this runtime only supports /v1/audio/speech")
        if request.model and request.model != self._client.model_name:
            raise ModelNotFoundError(
                f"model {request.model!r} is not served by this process")
        from ..media.media_source import (enforce_local_media_policy,
                                          message_media_modalities)

        unsupported_media = (message_media_modalities(request.messages) -
                             set(capabilities.input_modalities))
        if unsupported_media:
            requested = ", ".join(sorted(unsupported_media))
            supported = ", ".join(capabilities.input_modalities)
            raise UnsupportedFeatureError(
                f"the loaded model does not accept {requested} input; "
                f"supported input modalities: {supported}",
                param="messages",
            )

        try:
            enforce_local_media_policy(
                request.messages,
                self._config.allowed_local_media_path,
            )
        except PermissionError as exc:
            raise ServerError(str(exc), status_code=403) from exc
        if request.frequency_penalty != 0:
            raise UnsupportedFeatureError(
                "frequency_penalty is not supported by the Edge-LLM runtime",
                param="frequency_penalty")
        if request.presence_penalty != 0:
            raise UnsupportedFeatureError(
                "presence_penalty is not supported by the Edge-LLM runtime",
                param="presence_penalty")
        if request.seed is not None:
            raise UnsupportedFeatureError(
                "seed is not supported by the Edge-LLM runtime", param="seed")
        if request.response_format is not None:
            raise UnsupportedFeatureError(
                "response_format requires structured decoding, which is not "
                "implemented",
                param="response_format")

        try:
            tool_config = validate_tool_request(request.messages,
                                                request.tools,
                                                request.tool_choice,
                                                request.parallel_tool_calls)
        except ValueError as exc:
            raise InvalidRequestError(str(exc), param="tools") from exc
        if (tool_config.parse_output
                and tool_config.tool_choice in {"auto", "required"}
                and not self._config.enable_auto_tool_choice):
            raise UnsupportedFeatureError(
                "automatic tool choice is disabled; launch with "
                "--enable-auto-tool-choice",
                param="tool_choice")

        audio_params = None
        if request.audio is not None:
            if not capabilities.speech:
                raise UnsupportedFeatureError(
                    "audio output requires an Omni engine bundle",
                    param="modalities")
            if tool_config.parse_output:
                raise UnsupportedFeatureError(
                    "tools cannot be combined with audio output",
                    param="tools")
            if request.logprobs:
                raise UnsupportedFeatureError(
                    "logprobs cannot be combined with audio output",
                    param="logprobs")
            voices = self._client.llm.list_voices()
            if (request.audio.voice and voices
                    and request.audio.voice not in voices):
                raise InvalidRequestError(
                    f"unknown voice {request.audio.voice!r}; available: " +
                    ", ".join(voices),
                    param="audio.voice",
                )
            audio_params = AudioParams(**request.audio.generation_kwargs())

        reasoning_parser = (self._config.reasoning_parser
                            if request.enable_thinking else "none")
        try:
            parser = REASONING_PARSERS.resolve(reasoning_parser,
                                               self._model_dir)
        except KeyError as exc:
            raise InvalidRequestError(str(exc),
                                      param="reasoning_parser") from exc

        from ..runtime.engine import _normalize_logit_bias

        try:
            logit_bias = _normalize_logit_bias(request.logit_bias)
        except ValueError as exc:
            raise InvalidRequestError(str(exc), param="logit_bias") from exc

        num_logprobs = 0
        if request.logprobs:
            num_logprobs = max(1, request.top_logprobs or 0)
        greedy = request.temperature == 0
        sampling = SamplingParams(
            temperature=request.temperature,
            top_p=1.0 if greedy else request.top_p,
            top_k=1 if greedy else request.top_k,
            max_tokens=request.effective_max_tokens,
            enable_thinking=request.enable_thinking,
            disable_spec_decode=request.disable_spec_decode,
            num_logprobs=num_logprobs,
            stop=request.stop_strings,
            logit_bias=logit_bias,
            skip_special_tokens=(parser is None
                                 and not tool_config.parse_output),
            reuse_context=request.reuse_context,
            cache_generated_tokens=request.cache_generated_tokens,
        )
        return PreparedChatRequest(sampling, tool_config, reasoning_parser,
                                   audio_params)

    def parse_output(self, text: str, prepared: PreparedChatRequest):
        return parse_assistant_output(
            text,
            prepared.tool_config,
            self._model_dir,
            tool_parser=self._config.tool_call_parser,
            reasoning_parser=prepared.reasoning_parser,
        )

    def stream_output_parser(
            self,
            prepared: PreparedChatRequest) -> StreamingAssistantOutputParser:
        return stream_assistant_output(
            prepared.tool_config,
            self._model_dir,
            tool_parser=self._config.tool_call_parser,
            reasoning_parser=prepared.reasoning_parser,
        )

    async def create_chat_completion(
            self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        prepared = self.prepare_request(request)
        engine_request = await self.prepare_engine_request(request, prepared)
        if prepared.audio_params is not None:
            return await self._create_audio_completion(request, prepared,
                                                       engine_request)
        try:
            output = await self._client.generate(
                request.messages,
                prepared.sampling,
                tools=prepared.tool_config.tools,
                tool_choice=prepared.tool_config.tool_choice,
                tool_config=prepared.tool_config,
                tool_parser=self._config.tool_call_parser,
                reasoning_parser=prepared.reasoning_parser,
                prepared=engine_request,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidRequestError(str(exc)) from exc

        output.text = output.text.replace(IM_END_TOKEN, "")
        message = ChatCompletionMessage(
            content=(output.text or None)
            if output.tool_calls else output.text,
            reasoning_content=output.reasoning or None,
            tool_calls=output.tool_calls or None,
        )
        finish_reason = output.finish_reason or "stop"
        if output.tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"
        logprobs = None
        if request.logprobs:
            logprobs = _format_logprob_steps(
                output.token_ids,
                output.logprobs,
                request.top_logprobs is not None,
            )
        return ChatCompletionResponse(
            model=self._client.model_name,
            choices=[
                ChatCompletionChoice(
                    message=message,
                    logprobs=logprobs,
                    finish_reason=finish_reason,
                )
            ],
            usage=_usage(output.prompt_tokens, len(output.token_ids)),
        )

    async def _create_audio_completion(
        self,
        request: ChatCompletionRequest,
        prepared: PreparedChatRequest,
        engine_request: PreparedRequest,
    ) -> ChatCompletionResponse:
        text_parts: List[str] = []
        audio_parts: List[bytes] = []
        completion_tokens = 0
        prompt_tokens = None
        finish_reason = "stop"
        async for delta in self._client.stream_with_audio(
                request.messages,
                prepared.sampling,
                prepared.audio_params,
                prepared=engine_request):
            if delta.text:
                text_parts.append(delta.text)
            if delta.audio_bytes:
                audio_parts.append(delta.audio_bytes)
            completion_tokens += len(delta.token_ids)
            if delta.prompt_tokens is not None:
                prompt_tokens = delta.prompt_tokens
            if delta.finished:
                finish_reason = delta.finish_reason or "stop"

        parsed = self.parse_output(
            "".join(text_parts).replace(IM_END_TOKEN, ""), prepared)
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        message = ChatCompletionMessage(
            content=parsed.content,
            reasoning_content=parsed.reasoning or None,
            audio={
                "id": f"audio-{response_id}",
                "data": base64.b64encode(b"".join(audio_parts)).decode(),
                "format": "pcm16",
                "sample_rate": OMNI_AUDIO_SAMPLE_RATE,
                "transcript": parsed.content,
            },
        )
        return ChatCompletionResponse(
            id=response_id,
            model=self._client.model_name,
            choices=[
                ChatCompletionChoice(message=message,
                                     finish_reason=finish_reason)
            ],
            usage=_usage(prompt_tokens, completion_tokens),
        )

    async def prepare_engine_request(
        self,
        request: ChatCompletionRequest,
        prepared: PreparedChatRequest,
    ) -> PreparedRequest:
        engine_request = None
        try:
            engine_request = await self._client.prepare_request(
                request.messages,
                prepared.sampling,
                tools=prepared.tool_config.tools,
                tool_choice=prepared.tool_config.tool_choice,
                tool_config=prepared.tool_config,
            )
            return engine_request
        except (KeyError, TypeError, ValueError) as exc:
            if engine_request is not None:
                engine_request.release()
            raise InvalidRequestError(str(exc)) from exc
        except BaseException:
            if engine_request is not None:
                engine_request.release()
            raise

    async def stream_chat_completion(
        self,
        request: ChatCompletionRequest,
        prepared: Optional[PreparedChatRequest] = None,
        engine_stream: Optional[PreparedRequest] = None,
    ) -> AsyncGenerator[str, None]:
        prepared = prepared or self.prepare_request(request)
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        include_usage = bool(request.stream_options
                             and request.stream_options.include_usage)
        prompt_tokens = None

        yield self._chunk(response_id, created, DeltaMessage(role="assistant"))
        if prepared.audio_params is not None:
            async for chunk in self._stream_audio(request, prepared,
                                                  response_id, created,
                                                  include_usage,
                                                  engine_stream):
                yield chunk
            return
        if prepared.tool_config.parse_output:
            async for chunk in self._stream_tools(request, prepared,
                                                  response_id, created,
                                                  include_usage,
                                                  engine_stream):
                yield chunk
            return

        parser = REASONING_PARSERS.resolve(prepared.reasoning_parser,
                                           self._model_dir)
        stream_parser = parser.stream() if parser else None
        completion_tokens = 0
        finish_reason = "stop"
        try:
            async for delta in self._client.stream(
                    request.messages,
                    prepared.sampling,
                    tools=prepared.tool_config.tools,
                    tool_choice=prepared.tool_config.tool_choice,
                    prepared=engine_stream):
                completion_tokens += len(delta.token_ids)
                if delta.prompt_tokens is not None:
                    prompt_tokens = delta.prompt_tokens
                logprobs = _format_logprob_steps(
                    delta.token_ids,
                    delta.logprobs,
                    request.top_logprobs is not None,
                ) if request.logprobs else None
                if delta.text:
                    text = delta.text.replace(IM_END_TOKEN, "")
                    if stream_parser:
                        for parsed in stream_parser.feed(text):
                            message = DeltaMessage(
                                **({
                                    "reasoning_content": parsed.text
                                } if parsed.field == "reasoning" else {
                                    "content": parsed.text
                                }))
                            yield self._chunk(
                                response_id,
                                created,
                                message,
                                logprobs=logprobs,
                            )
                            logprobs = None
                    else:
                        yield self._chunk(
                            response_id,
                            created,
                            DeltaMessage(content=text),
                            logprobs=logprobs,
                        )
                        logprobs = None
                if logprobs is not None:
                    # Reached when the parser withheld every byte of this
                    # delta: its logprobs still ship, on an empty chunk.
                    yield self._chunk(response_id,
                                      created,
                                      DeltaMessage(),
                                      logprobs=logprobs)
                if delta.finished:
                    finish_reason = delta.finish_reason or "stop"
        except asyncio.CancelledError:
            raise
        except ServerError as exc:
            logger.warning("Streaming request failed: %s", exc)
            yield _sse_error(exc)
            yield "data: [DONE]\n\n"
            return
        except (KeyError, TypeError, ValueError) as exc:
            error = InvalidRequestError(str(exc))
            logger.warning("Streaming request failed: %s", error)
            yield _sse_error(error)
            yield "data: [DONE]\n\n"
            return

        if stream_parser:
            for parsed in stream_parser.flush():
                delta = ({
                    "reasoning_content": parsed.text
                } if parsed.field == "reasoning" else {
                    "content": parsed.text
                })
                yield self._chunk(response_id, created, DeltaMessage(**delta))
        yield self._chunk(response_id,
                          created,
                          DeltaMessage(),
                          finish_reason=finish_reason)
        if include_usage:
            usage_chunk = ChatCompletionStreamResponse(
                id=response_id,
                created=created,
                model=self._client.model_name,
                choices=[],
                usage=_usage(prompt_tokens, completion_tokens),
            )
            yield _sse(usage_chunk)
        yield "data: [DONE]\n\n"

    async def _stream_audio(
        self,
        request: ChatCompletionRequest,
        prepared: PreparedChatRequest,
        response_id: str,
        created: int,
        include_usage: bool,
        engine_request: Optional[PreparedRequest],
    ) -> AsyncGenerator[str, None]:
        parser = REASONING_PARSERS.resolve(prepared.reasoning_parser,
                                           self._model_dir)
        stream_parser = parser.stream() if parser else None
        completion_tokens = 0
        prompt_tokens = None
        finish_reason = "stop"
        audio_id = f"audio-{response_id}"
        try:
            async for delta in self._client.stream_with_audio(
                    request.messages,
                    prepared.sampling,
                    prepared.audio_params,
                    prepared=engine_request):
                completion_tokens += len(delta.token_ids)
                if delta.prompt_tokens is not None:
                    prompt_tokens = delta.prompt_tokens
                if delta.text:
                    text = delta.text.replace(IM_END_TOKEN, "")
                    if stream_parser:
                        for parsed in stream_parser.feed(text):
                            field = ("reasoning_content" if parsed.field
                                     == "reasoning" else "content")
                            yield self._chunk(
                                response_id,
                                created,
                                DeltaMessage(**{field: parsed.text}),
                            )
                    else:
                        yield self._chunk(response_id, created,
                                          DeltaMessage(content=text))
                if delta.audio_bytes:
                    yield self._chunk(
                        response_id,
                        created,
                        DeltaMessage(
                            audio={
                                "id":
                                audio_id,
                                "data":
                                base64.b64encode(delta.audio_bytes).decode(),
                                "format":
                                "pcm16",
                                "sample_rate":
                                OMNI_AUDIO_SAMPLE_RATE,
                            }),
                    )
                if delta.finished:
                    finish_reason = delta.finish_reason or "stop"
        except asyncio.CancelledError:
            raise
        except ServerError as exc:
            yield _sse_error(exc)
            yield "data: [DONE]\n\n"
            return

        if stream_parser:
            for parsed in stream_parser.flush():
                field = ("reasoning_content"
                         if parsed.field == "reasoning" else "content")
                yield self._chunk(response_id, created,
                                  DeltaMessage(**{field: parsed.text}))
        yield self._chunk(response_id,
                          created,
                          DeltaMessage(),
                          finish_reason=finish_reason)
        if include_usage:
            yield _sse(
                ChatCompletionStreamResponse(
                    id=response_id,
                    created=created,
                    model=self._client.model_name,
                    choices=[],
                    usage=_usage(prompt_tokens, completion_tokens),
                ))
        yield "data: [DONE]\n\n"

    async def _stream_tools(
        self,
        request: ChatCompletionRequest,
        prepared: PreparedChatRequest,
        response_id: str,
        created: int,
        include_usage: bool,
        engine_stream: Optional[PreparedRequest],
    ) -> AsyncGenerator[str, None]:
        parser = _select_parser(self._model_dir,
                                self._config.tool_call_parser).stream(
                                    prepared.tool_config)
        reasoning = REASONING_PARSERS.resolve(prepared.reasoning_parser,
                                              self._model_dir)
        reasoning_stream = reasoning.stream() if reasoning else None
        completion_tokens = 0
        prompt_tokens = None
        finish_reason = "stop"
        tool_heads = 0

        def messages_for(events):
            nonlocal tool_heads
            for event in events:
                if event.kind == "content":
                    if reasoning_stream is not None:
                        for parsed in reasoning_stream.feed(event.text):
                            field = ("reasoning_content" if parsed.field
                                     == "reasoning" else "content")
                            yield DeltaMessage(**{field: parsed.text})
                    elif event.text:
                        yield DeltaMessage(content=event.text)
                elif event.kind == "tool_head":
                    tool_heads += 1
                    yield DeltaMessage(tool_calls=[{
                        "index": event.index,
                        "id": event.call_id,
                        "type": "function",
                        "function": {
                            "name": event.name,
                            "arguments": "",
                        },
                    }])
                elif event.kind == "tool_args":
                    yield DeltaMessage(tool_calls=[{
                        "index": event.index,
                        "function": {
                            "arguments": event.text,
                        },
                    }])

        try:
            async for delta in self._client.stream(
                    request.messages,
                    prepared.sampling,
                    tools=prepared.tool_config.tools,
                    tool_choice=prepared.tool_config.tool_choice,
                    prepared=engine_stream):
                completion_tokens += len(delta.token_ids)
                if delta.prompt_tokens is not None:
                    prompt_tokens = delta.prompt_tokens
                logprobs = _format_logprob_steps(
                    delta.token_ids,
                    delta.logprobs,
                    request.top_logprobs is not None,
                ) if request.logprobs else None
                if delta.text:
                    text = delta.text.replace(IM_END_TOKEN, "")
                    for message in messages_for(parser.feed(text)):
                        yield self._chunk(response_id,
                                          created,
                                          message,
                                          logprobs=logprobs)
                        logprobs = None
                if logprobs is not None:
                    # Reached when the parser withheld every byte of this
                    # delta: its logprobs still ship, on an empty chunk.
                    yield self._chunk(response_id,
                                      created,
                                      DeltaMessage(),
                                      logprobs=logprobs)
                if delta.finished:
                    finish_reason = delta.finish_reason or "stop"
        except asyncio.CancelledError:
            raise
        except ServerError as exc:
            logger.warning("Streaming tool request failed: %s", exc)
            yield _sse_error(exc)
            yield "data: [DONE]\n\n"
            return
        except (KeyError, TypeError, ValueError) as exc:
            error = InvalidRequestError(str(exc))
            logger.warning("Streaming tool request failed: %s", error)
            yield _sse_error(error)
            yield "data: [DONE]\n\n"
            return

        for message in messages_for(parser.flush()):
            yield self._chunk(response_id, created, message)
        if reasoning_stream is not None:
            for parsed in reasoning_stream.flush():
                field = ("reasoning_content"
                         if parsed.field == "reasoning" else "content")
                yield self._chunk(response_id, created,
                                  DeltaMessage(**{field: parsed.text}))

        if tool_heads and finish_reason == "stop":
            finish_reason = "tool_calls"
        yield self._chunk(response_id,
                          created,
                          DeltaMessage(),
                          finish_reason=finish_reason)
        if include_usage:
            usage_chunk = ChatCompletionStreamResponse(
                id=response_id,
                created=created,
                model=self._client.model_name,
                choices=[],
                usage=_usage(prompt_tokens, completion_tokens),
            )
            yield _sse(usage_chunk)
        yield "data: [DONE]\n\n"

    def _chunk(
        self,
        response_id: str,
        created: int,
        delta: DeltaMessage,
        *,
        finish_reason: Optional[str] = None,
        logprobs: Optional[Dict[str, Any]] = None,
    ) -> str:
        response = ChatCompletionStreamResponse(
            id=response_id,
            created=created,
            model=self._client.model_name,
            choices=[
                ChatCompletionStreamChoice(
                    delta=delta,
                    logprobs=logprobs,
                    finish_reason=finish_reason,
                )
            ],
        )
        return _sse(response)
