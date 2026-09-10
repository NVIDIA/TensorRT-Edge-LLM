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
"""FastAPI application assembly and cross-cutting server middleware."""

import hmac
import logging
from contextlib import asynccontextmanager
from typing import Awaitable, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from tensorrt_edgellm import __version__

from ..config import ApiConfig
from ..media.video_sampling import MAX_SOURCE_BYTES as MAX_VIDEO_SOURCE_BYTES
from ..runtime.engine_client import EngineClient
from .errors import ServerError
from .routes import router
from .serving_anthropic import AnthropicServingMessages
from .serving_audio import (MAX_AUDIO_UPLOAD_BYTES, OpenAIServingSpeech,
                            OpenAIServingTranscription)
from .serving_chat import OpenAIServingChat

logger = logging.getLogger("edgellm.server")
MAX_REQUEST_BODY_BYTES = (-(-MAX_VIDEO_SOURCE_BYTES // 3) * 4 + 1024 * 1024)


def _content_length(value: Optional[str]) -> Optional[int]:
    """Parse a Content-Length header without accepting partial numbers."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return -1


def _lifespan(engine_client: EngineClient):
    """Create an application lifespan that owns the engine client."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Release native runtime resources after request processing stops."""
        try:
            yield
        finally:
            await engine_client.close()

    return lifespan


def _configure_services(app: FastAPI, engine_client: EngineClient,
                        config: ApiConfig) -> None:
    """Attach the shared runtime and protocol-specific serving adapters."""
    app.state.engine_client = engine_client
    app.state.server_config = config
    app.state.openai_serving_chat = OpenAIServingChat(engine_client, config)
    app.state.anthropic_serving_messages = AnthropicServingMessages(
        engine_client, app.state.openai_serving_chat)
    app.state.openai_serving_transcription = OpenAIServingTranscription(
        engine_client)
    app.state.openai_serving_speech = OpenAIServingSpeech(engine_client)


def _error_response(request: Request, error: ServerError) -> JSONResponse:
    """Render an error using the wire contract selected by the route."""
    if request.url.path.startswith("/v1/messages"):
        from .anthropic_compat import error_payload

        status = 529 if error.status_code == 429 else error.status_code
        return JSONResponse(status_code=status,
                            content=error_payload(status, str(error)))
    headers = ({"Retry-After": "1"} if error.status_code == 429 else None)
    return JSONResponse(status_code=error.status_code,
                        content=error.payload(),
                        headers=headers)


def _register_exception_handlers(app: FastAPI) -> None:
    """Register protocol-aware handlers for expected and unexpected errors."""

    @app.exception_handler(ServerError)
    async def _server_error_handler(request: Request, exc: ServerError):
        """Return an explicitly classified server error."""
        return _error_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(request: Request,
                                        exc: RequestValidationError):
        """Convert FastAPI schema failures to the public error contract."""
        first = exc.errors()[0] if exc.errors() else {}
        location = first.get("loc", ())
        param = ".".join(str(item) for item in location[1:]) or None
        message = first.get("msg", "invalid request")
        error = ServerError(str(message), param=param)
        return _error_response(request, error)

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(request: Request, exc: Exception):
        """Hide internal exception details while retaining a traceback."""
        logger.error("Unhandled server error",
                     exc_info=(type(exc), exc, exc.__traceback__))
        error = ServerError(
            "internal server error",
            status_code=500,
            error_type="internal_server_error",
        )
        return _error_response(request, error)


def _authentication_error(request: Request,
                          config: ApiConfig) -> Optional[JSONResponse]:
    """Authenticate a versioned API request when a server key is configured."""
    if not config.api_key or not request.url.path.startswith("/v1/"):
        return None
    expected = f"Bearer {config.api_key}"
    bearer = request.headers.get("authorization", "")
    anthropic_key = request.headers.get("x-api-key", "")
    if (hmac.compare_digest(bearer, expected)
            or hmac.compare_digest(anthropic_key, config.api_key)):
        return None
    error = ServerError(
        "invalid API key",
        status_code=401,
        error_type="authentication_error",
    )
    response = _error_response(request, error)
    response.headers["WWW-Authenticate"] = "Bearer"
    return response


def _audio_size_error(request: Request) -> Optional[JSONResponse]:
    """Validate upload size before multipart parsing allocates the body."""
    length = _content_length(request.headers.get("content-length"))
    if length is None:
        error = ServerError("Content-Length is required",
                            status_code=411,
                            error_type="invalid_request_error")
        return _error_response(request, error)
    if length < 0:
        return _error_response(request,
                               ServerError("invalid Content-Length header"))
    if length > MAX_AUDIO_UPLOAD_BYTES + 1024 * 1024:
        error = ServerError("audio upload exceeds the 25 MiB limit",
                            status_code=413,
                            error_type="invalid_request_error")
        return _error_response(request, error)
    return None


async def _buffer_request_body(request: Request) -> Optional[JSONResponse]:
    """Buffer a bounded JSON request so downstream validation can reread it."""
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_REQUEST_BODY_BYTES:
            error = ServerError(
                "request body exceeds the supported maximum",
                status_code=413,
                error_type="invalid_request_error",
            )
            return _error_response(request, error)
    request._body = bytes(body)
    return None


def _register_request_guards(app: FastAPI, config: ApiConfig) -> None:
    """Register authentication and request-size guards."""

    @app.middleware("http")
    async def _server_guards(
        request: Request,
        call_next: Callable[[Request], Awaitable],
    ):
        """Reject unauthorized or oversized requests before route handling."""
        error_response = _authentication_error(request, config)
        if error_response is not None:
            return error_response
        if request.url.path == "/v1/audio/transcriptions":
            error_response = _audio_size_error(request)
        elif request.method == "POST":
            error_response = await _buffer_request_body(request)
        if error_response is not None:
            return error_response
        return await call_next(request)


def create_app(engine_client: EngineClient,
               config: Optional[ApiConfig] = None) -> FastAPI:
    """Create the HTTP application around one loaded engine client."""
    config = config or ApiConfig()
    app = FastAPI(
        title="TensorRT Edge-LLM Server",
        version=__version__,
        description="OpenAI-compatible inference powered by TensorRT Edge-LLM",
        lifespan=_lifespan(engine_client),
    )
    _configure_services(app, engine_client, config)
    _register_exception_handlers(app)
    _register_request_guards(app, config)

    app.include_router(router)
    return app


def run_http_server(engine_client: EngineClient, config: ApiConfig) -> None:
    """Run the application with the optional Uvicorn server dependency."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is required; install tensorrt-edgellm[server]") from exc
    uvicorn.run(
        create_app(engine_client, config),
        host=config.host,
        port=config.port,
        log_level=config.log_level,
    )
