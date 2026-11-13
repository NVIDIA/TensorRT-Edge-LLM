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
"""FastAPI routes for OpenAI and Anthropic protocol adapters."""

from typing import Any, Dict

import anyio
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import (JSONResponse, PlainTextResponse,
                               StreamingResponse)
from pydantic import ValidationError

from . import anthropic_compat, responses_compat
from .errors import ServerError
from .protocol import (ChatCompletionRequest, CompletionRequest, ModelCard,
                       ModelList, SpeechRequest, TranscriptionResponse)

router = APIRouter()


class _ReleasingStreamingResponse(StreamingResponse):

    def __init__(self, *args, prepared_stream, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepared_stream = prepared_stream

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    close = getattr(self.body_iterator, "aclose", None)
                    if close is not None:
                        await close()
                finally:
                    self._prepared_stream.release()


class _SseResponse(_ReleasingStreamingResponse):
    """An SSE response whose headers go out with its first chunk.

    A ServerError raised before that chunk reaches the app's exception
    handler, so the client gets the failure's HTTP status and JSON body, as
    without streaming. Later failures are error events in the stream.
    """

    def __init__(self, content, *, prepared_stream):
        super().__init__(content,
                         prepared_stream=prepared_stream,
                         media_type="text/event-stream",
                         headers={
                             "Cache-Control": "no-cache",
                             "Connection": "keep-alive",
                         })

    async def __call__(self, scope, receive, send):
        started = False

        async def send_and_track(message):
            nonlocal started
            started = True
            await send(message)

        await super().__call__(scope, receive, send_and_track)
        if not started:
            # The client left before the first chunk. ASGI middleware still
            # expects a response.
            await send({
                "type": "http.response.start",
                "status": 499,
                "headers": []
            })
            await send({"type": "http.response.body", "body": b""})

    async def stream_response(self, send):
        first = await anext(self.body_iterator, None)
        if first is not None:
            self.body_iterator = _prepend(first, self.body_iterator)
        await super().stream_response(send)


async def _prepend(first, rest):
    try:
        yield first
        async for chunk in rest:
            yield chunk
    finally:
        await rest.aclose()


@router.get("/health")
async def health(request: Request):
    client = request.app.state.engine_client
    caps = client.capabilities
    scheduling = None
    getter = getattr(client.llm, "get_scheduling_metrics", None)
    if getter is not None:
        scheduling = getter()
    context_cache = None
    getter = getattr(client.llm, "get_context_cache_counters", None)
    if getter is not None:
        context_cache = getter()
    engine = client.engine_status()
    healthy = engine is None or engine["serving"]
    body = {
        "status": "healthy" if healthy else "unhealthy",
        "model": client.model_name,
        "active_requests": client.active_requests,
        "queued_requests": client.queued_requests,
        "scheduling": scheduling,
        "context_cache": context_cache,
        "engine": engine,
        "capabilities": {
            "chat": caps.chat,
            "transcription": caps.transcription,
            "speech": caps.speech,
            "input_modalities": list(caps.input_modalities),
            "output_modalities": list(caps.output_modalities),
            "max_model_len": caps.max_model_len,
            "max_input_len": caps.max_input_len,
            "max_batch_size": caps.max_batch_size,
            "max_num_seqs": caps.max_num_seqs,
            "kv_cache_dtype": caps.kv_cache_dtype,
            "speculative_decoding": caps.speculative_decoding,
            "speculative_method": caps.speculative_method,
            "context_reuse": caps.context_reuse,
            "in_flight_batching": caps.in_flight_batching,
        },
    }
    if not healthy:
        # The engine fails every request until the server restarts; 503 lets a liveness probe act on it.
        body["reason"] = engine["unavailable_reason"]
        return JSONResponse(status_code=503, content=body)
    return body


@router.get("/health/ready")
async def readiness(request: Request):
    client = request.app.state.engine_client
    engine = client.engine_status()
    if engine is not None and not engine["serving"]:
        return JSONResponse(status_code=503,
                            content={
                                "status": "unavailable",
                                "reason": engine["unavailable_reason"],
                                "active_requests": client.active_requests,
                                "queued_requests": client.queued_requests,
                            })
    return {
        "status": "ready",
        "active_requests": client.active_requests,
        "queued_requests": client.queued_requests,
    }


@router.get("/v1/models")
async def list_models(request: Request):
    client = request.app.state.engine_client
    response = ModelList(data=[
        ModelCard(
            id=client.model_name,
            max_model_len=client.capabilities.max_model_len,
        )
    ])
    return JSONResponse(response.model_dump(exclude_none=True))


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    handler = request.app.state.openai_serving_chat
    if body.stream:
        prepared = handler.prepare_request(body)
        engine_stream = await handler.prepare_engine_request(body, prepared)
        return _SseResponse(handler.stream_chat_completion(
            body, prepared, engine_stream),
                            prepared_stream=engine_stream)
    response = await handler.create_chat_completion(body)
    return JSONResponse(response.model_dump(exclude_none=True))


@router.post("/v1/responses")
async def responses(body: Dict[str, Any], request: Request):
    handler = request.app.state.openai_serving_chat
    try:
        chat_fields = responses_compat.to_chat_request(body)
        streaming = bool(body.get("stream"))
        if streaming:
            chat_fields["stream"] = True
        chat_body = ChatCompletionRequest(**chat_fields)
        if streaming:
            prepared = handler.prepare_request(chat_body)
            engine_stream = await handler.prepare_engine_request(
                chat_body, prepared)
        else:
            completion = await handler.create_chat_completion(chat_body)
    except (ServerError, TypeError, ValueError, ValidationError) as exc:
        status = exc.status_code if isinstance(exc, ServerError) else 400
        return JSONResponse(status_code=status,
                            content={
                                "error": {
                                    "message": str(exc),
                                    "type": "invalid_request_error",
                                }
                            })
    if streaming:

        async def generate():
            translator = responses_compat.ResponsesStream(chat_body.model)
            # Sent with the first chat chunk, which waits for generation to start.
            pending = translator.start()
            async for chunk in handler.stream_chat_completion(
                    chat_body, prepared, engine_stream):
                for event in pending + translator.push_sse(chunk):
                    yield event
                pending = []
            for event in translator.finish():
                yield event

        return _SseResponse(generate(), prepared_stream=engine_stream)
    return JSONResponse(
        responses_compat.to_response_object(
            completion.model_dump(exclude_none=True)))


@router.post("/v1/completions")
async def completions(body: CompletionRequest, request: Request):
    handler = request.app.state.openai_serving_completion
    if body.stream:
        prepared = handler.prepare_request(body)
        engine_stream = await handler.prepare_engine_request(body, prepared)
        return _SseResponse(handler.stream_completion(body, prepared,
                                                      engine_stream),
                            prepared_stream=engine_stream)
    response = await handler.create_completion(body)
    return JSONResponse(response.model_dump(exclude_none=True))


def _anthropic_error(exc: Exception) -> JSONResponse:
    status = exc.status_code if isinstance(exc, ServerError) else 400
    if status == 429:
        status = 529
    return JSONResponse(status_code=status,
                        content=anthropic_compat.error_payload(
                            status, str(exc)))


@router.post("/v1/messages")
async def anthropic_messages(body: Dict[str, Any], request: Request):
    handler = request.app.state.anthropic_serving_messages
    try:
        if body.get("stream"):
            prepared = await handler.prepare_stream(body)
            return _SseResponse(handler.stream_message(prepared),
                                prepared_stream=prepared.engine)
        return JSONResponse(await handler.create_message(body))
    except (TypeError, ValueError) as exc:
        # A ServerError reaches the app's handler, which renders the Anthropic body with its Retry-After header.
        return _anthropic_error(exc)


@router.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(body: Dict[str, Any], request: Request):
    handler = request.app.state.anthropic_serving_messages
    try:
        return {"input_tokens": await handler.count_tokens(body)}
    except (ServerError, TypeError, ValueError) as exc:
        return _anthropic_error(exc)


@router.post("/v1/audio/transcriptions")
async def audio_transcriptions(
        request: Request,
        file: UploadFile = File(...),
        model: str = Form(""),
        prompt: str = Form(""),
        language: str = Form(""),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
):
    if response_format not in {"json", "text"}:
        from .errors import InvalidRequestError
        raise InvalidRequestError(
            "response_format must be json or text",
            param="response_format",
        )
    handler = request.app.state.openai_serving_transcription
    result: TranscriptionResponse = await handler.transcribe(
        file,
        model=model,
        prompt=prompt,
        language=language,
        temperature=temperature,
    )
    if response_format == "text":
        return PlainTextResponse(result.text)
    return JSONResponse(result.model_dump(exclude_none=True))


@router.get("/v1/voices")
async def list_voices(request: Request):
    client = request.app.state.engine_client
    voices = client.llm.list_voices() if client.capabilities.speech else []
    return {
        "object": "list",
        "data": [{
            "id": voice,
            "object": "voice"
        } for voice in voices],
    }


@router.post("/v1/audio/speech")
async def audio_speech(body: SpeechRequest, request: Request):
    handler = request.app.state.openai_serving_speech
    prepared = await handler.prepare_stream(body)
    return _ReleasingStreamingResponse(
        handler.stream(body, prepared=prepared),
        prepared_stream=prepared,
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": "24000",
            "X-Channels": "1",
            "X-Sample-Format": "s16le",
        },
    )
