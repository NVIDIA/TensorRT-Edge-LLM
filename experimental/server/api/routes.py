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

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import (JSONResponse, PlainTextResponse,
                               StreamingResponse)

from . import anthropic_compat
from .errors import ServerError
from .protocol import (ChatCompletionRequest, ModelCard, ModelList,
                       SpeechRequest, TranscriptionResponse)

router = APIRouter()


class _ReleasingStreamingResponse(StreamingResponse):

    def __init__(self, *args, prepared_stream, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepared_stream = prepared_stream

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._prepared_stream.release()


@router.get("/health")
async def health(request: Request):
    client = request.app.state.engine_client
    caps = client.capabilities
    return {
        "status": "healthy",
        "model": client.model_name,
        "active_requests": client.active_requests,
        "queued_requests": client.queued_requests,
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
        },
    }


@router.get("/health/ready")
async def readiness(request: Request):
    client = request.app.state.engine_client
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
        return _ReleasingStreamingResponse(
            handler.stream_chat_completion(body, prepared, engine_stream),
            prepared_stream=engine_stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    response = await handler.create_chat_completion(body)
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
            return _ReleasingStreamingResponse(
                handler.stream_message(prepared),
                prepared_stream=prepared.engine,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        return JSONResponse(await handler.create_message(body))
    except (ServerError, TypeError, ValueError) as exc:
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
    # Validate before response headers are sent.
    handler.validate(body)
    return StreamingResponse(
        handler.stream(body),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": "24000",
            "X-Channels": "1",
            "X-Sample-Format": "s16le",
        },
    )
