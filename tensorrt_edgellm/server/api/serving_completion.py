# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""OpenAI text-completion serving for raw text and token-ID prompts."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

from ..runtime.engine import SamplingParams
from ..runtime.engine_client import EngineClient, PreparedRequest
from . import serving_utils
from .errors import (InvalidRequestError, ModelNotFoundError, ServerError,
                     UnsupportedFeatureError)
from .protocol import (CompletionChoice, CompletionRequest, CompletionResponse,
                       CompletionStreamResponse)

logger = logging.getLogger("edgellm.server.completion")


@dataclass(frozen=True)
class PreparedCompletionRequest:
    sampling: SamplingParams
    prompt_tokens: Optional[int]


class OpenAIServingCompletion:
    """Translate OpenAI completion requests to the shared engine client."""

    def __init__(self, engine_client: EngineClient) -> None:
        self._client = engine_client

    def prepare_request(
            self, request: CompletionRequest) -> PreparedCompletionRequest:
        if not self._client.capabilities.chat:
            raise UnsupportedFeatureError(
                "this runtime does not support /v1/completions")
        if request.model and request.model != self._client.model_name:
            raise ModelNotFoundError(
                f"model {request.model!r} is not served by this process")
        greedy = request.temperature == 0
        sampling = SamplingParams(
            temperature=request.temperature,
            top_p=1.0 if greedy else request.top_p,
            top_k=1 if greedy else request.top_k,
            max_tokens=request.max_tokens,
            stop=request.stop_strings,
            disable_spec_decode=request.disable_spec_decode,
            ignore_eos=request.ignore_eos,
            skip_special_tokens=True,
            reuse_context=request.reuse_context,
            cache_generated_tokens=request.cache_generated_tokens,
            priority=request.priority,
        )
        serving_utils.check_cache_generated_tokens(sampling, self._client.llm)
        return PreparedCompletionRequest(sampling, request.prompt_token_count)

    async def prepare_engine_request(
        self,
        request: CompletionRequest,
        prepared: PreparedCompletionRequest,
    ) -> PreparedRequest:
        try:
            return await self._client.prepare_completion(
                request.prompt, prepared.sampling)
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidRequestError(str(exc), param="prompt") from exc

    async def create_completion(
            self, request: CompletionRequest) -> CompletionResponse:
        prepared = self.prepare_request(request)
        engine_request = await self.prepare_engine_request(request, prepared)
        try:
            output = await self._client.generate([],
                                                 prepared.sampling,
                                                 prepared=engine_request)
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidRequestError(str(exc)) from exc
        prompt_tokens = (prepared.prompt_tokens if prepared.prompt_tokens
                         is not None else output.prompt_tokens)
        return CompletionResponse(
            model=self._client.model_name,
            choices=[
                CompletionChoice(
                    text=output.text,
                    finish_reason=output.finish_reason or "stop",
                )
            ],
            usage=serving_utils.usage(prompt_tokens, len(output.token_ids),
                                      output.reused_prompt_tokens),
        )

    async def stream_completion(
        self,
        request: CompletionRequest,
        prepared: PreparedCompletionRequest,
        engine_stream: PreparedRequest,
    ) -> AsyncGenerator[str, None]:
        response_id = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        include_usage = bool(request.stream_options
                             and request.stream_options.include_usage)
        prompt_tokens = prepared.prompt_tokens
        reused_prompt_tokens = None
        completion_tokens = 0
        finish_reason = "stop"
        deltas = await serving_utils.await_first_output(
            self._client.stream([], prepared.sampling, prepared=engine_stream))
        try:
            async for delta in deltas:
                completion_tokens += len(delta.token_ids)
                if prompt_tokens is None and delta.prompt_tokens is not None:
                    prompt_tokens = delta.prompt_tokens
                if delta.reused_prompt_tokens is not None:
                    reused_prompt_tokens = delta.reused_prompt_tokens
                if delta.text or delta.token_ids:
                    yield self._chunk(response_id, created, delta.text)
                if delta.finished:
                    finish_reason = delta.finish_reason or "stop"
        except asyncio.CancelledError:
            raise
        except ServerError as exc:
            logger.warning("Streaming request failed: %s", exc)
            yield serving_utils.sse_error(exc)
            yield "data: [DONE]\n\n"
            return
        except (KeyError, TypeError, ValueError) as exc:
            error = InvalidRequestError(str(exc))
            logger.warning("Streaming request failed: %s", error)
            yield serving_utils.sse_error(error)
            yield "data: [DONE]\n\n"
            return

        yield self._chunk(response_id,
                          created,
                          "",
                          finish_reason=finish_reason)
        if include_usage:
            yield serving_utils.sse(
                CompletionStreamResponse(
                    id=response_id,
                    created=created,
                    model=self._client.model_name,
                    choices=[],
                    usage=serving_utils.usage(prompt_tokens, completion_tokens,
                                              reused_prompt_tokens),
                ))
        yield "data: [DONE]\n\n"

    def _chunk(
        self,
        response_id: str,
        created: int,
        text: str,
        *,
        finish_reason: Optional[str] = None,
    ) -> str:
        return serving_utils.sse(
            CompletionStreamResponse(
                id=response_id,
                created=created,
                model=self._client.model_name,
                choices=[
                    CompletionChoice(text=text, finish_reason=finish_reason)
                ],
            ))
