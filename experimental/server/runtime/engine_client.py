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
"""Async server boundary around the synchronous Edge-LLM high-level API."""

import asyncio
import json
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, Union

from ..api.errors import (EngineError, ServerError, ServerOverloadedError,
                          ServerUnavailableError, UnsupportedFeatureError)
from ..config import ApiConfig
from ..parsing.tool_calling import ToolConfig, validate_tool_request
from .engine import (LLM, TTS, AudioParams, CompletionOutput, SamplingParams,
                     StreamDelta)


@dataclass(frozen=True)
class EngineCapabilities:
    """Runtime facts exposed to routing, health, and model endpoints."""

    chat: bool
    transcription: bool
    speech: bool
    input_modalities: Sequence[str]
    output_modalities: Sequence[str]
    max_model_len: Optional[int]
    max_input_len: Optional[int]
    max_batch_size: Optional[int]
    max_num_seqs: int
    kv_cache_dtype: str
    speculative_decoding: bool
    speculative_method: str
    context_reuse: bool


class _AdmissionController:
    """Bounded queue for the runtime's single generation slot."""

    def __init__(self, max_queued_requests: int, timeout: float) -> None:
        self._semaphore = asyncio.Semaphore(1)
        self._max_queued = max_queued_requests
        self._timeout = timeout
        self._active = 0
        self._waiting = 0
        self._closing = False
        self._drained = False
        self._close_lock = asyncio.Lock()

    @property
    def active(self) -> int:
        return self._active

    @property
    def waiting(self) -> int:
        return self._waiting

    async def reserve(self) -> "_AdmissionLease":
        acquired = False
        if self._closing:
            raise ServerUnavailableError()
        if self._active + self._waiting >= self._max_queued + 1:
            raise ServerOverloadedError()
        self._waiting += 1

        try:
            try:
                await asyncio.wait_for(self._semaphore.acquire(),
                                       timeout=self._timeout)
                acquired = True
            except asyncio.TimeoutError as exc:
                raise ServerOverloadedError(
                    "timed out waiting for the Edge-LLM generation slot"
                ) from exc
            finally:
                self._waiting -= 1

            if self._closing:
                self._semaphore.release()
                acquired = False
                raise ServerUnavailableError()
            self._active = 1
            return _AdmissionLease(self)
        except BaseException:
            if acquired:
                self._semaphore.release()
            raise

    def release(self) -> None:
        self._active = 0
        self._semaphore.release()

    async def close(self) -> None:
        """Reject new work and wait until the active request has released."""
        async with self._close_lock:
            if self._drained:
                return
            self._closing = True
            await self._semaphore.acquire()
            self._drained = True


class _AdmissionLease:

    def __init__(self, controller: _AdmissionController) -> None:
        self._controller = controller
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._controller.release()


@dataclass
class PreparedRequest:
    """One native request and the generation lease owning its buffers."""

    request: Any
    lease: _AdmissionLease

    def release(self) -> None:
        self.lease.release()


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _capabilities_for(llm: Union[LLM, TTS]) -> EngineCapabilities:
    bundle_dir = llm.bundle_dir
    layout = llm.bundle_layout
    candidates = [
        os.path.join(bundle_dir, "config.json"),
        os.path.join(bundle_dir, "base_config.json"),
    ]
    config: Dict[str, Any] = {}
    for path in candidates:
        config = _read_json(path)
        if config:
            break
    builder = config.get("builder_config", {})
    if not isinstance(builder, dict):
        builder = {}

    chat = llm.runtime_kind == "chat"
    input_modalities = ["text"]
    if chat and layout.visual_dir:
        input_modalities.append("image")
        if llm.video_capable:
            input_modalities.append("video")
    if chat and layout.audio_dir:
        input_modalities.append("audio")
    output_modalities = ["text"] if chat else []
    if layout.has_speech:
        output_modalities.append("audio")

    max_kv = builder.get("max_kv_cache_capacity")
    max_positions = config.get("max_position_embeddings")
    max_model_len = max_kv if isinstance(max_kv, int) else max_positions
    return EngineCapabilities(
        chat=chat,
        transcription=chat and layout.has_transcription,
        speech=layout.has_speech,
        input_modalities=tuple(input_modalities),
        output_modalities=tuple(output_modalities),
        max_model_len=max_model_len
        if isinstance(max_model_len, int) else None,
        max_input_len=builder.get("max_input_len") if isinstance(
            builder.get("max_input_len"), int) else None,
        max_batch_size=builder.get("max_batch_size") if isinstance(
            builder.get("max_batch_size"), int) else None,
        max_num_seqs=1,
        kv_cache_dtype=str(config.get("kv_cache_dtype", "unknown")),
        speculative_decoding=llm.has_draft_model,
        speculative_method=str(config.get("spec_decode_type", "none")),
        context_reuse=bool(getattr(llm, "context_cache_enabled", False)),
    )


_STREAM_END = object()


def _next_stream_item(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return _STREAM_END


def _close_stream(iterator) -> None:
    try:
        iterator.close()
    except (RuntimeError, ValueError):
        # RuntimeError/ValueError means a final next() is still unwinding. The
        # caller waits for that future before reaching this helper.
        pass


async def _iterate_sync(iterator):
    """Advance a native stream and cancel it before waiting on disconnect."""
    while True:
        next_task = asyncio.create_task(
            asyncio.to_thread(_next_stream_item, iterator))
        try:
            item = await asyncio.shield(next_task)
        except asyncio.CancelledError:
            close_task = asyncio.create_task(
                asyncio.to_thread(_close_stream, iterator))
            await asyncio.shield(
                asyncio.gather(next_task, close_task, return_exceptions=True))
            raise
        if item is _STREAM_END:
            return
        yield item


async def _run_sync(operation):
    """Run blocking work without outliving the request that owns its lease."""
    worker = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        await asyncio.shield(asyncio.gather(worker, return_exceptions=True))
        raise


class EngineClient:
    """Asynchronous, bounded-queue adapter for one Edge-LLM runtime."""

    def __init__(self,
                 llm: Union[LLM, TTS],
                 api_config: Optional[ApiConfig] = None) -> None:
        self._llm = llm
        self._api_config = api_config or ApiConfig()
        model_dir = llm.model_dir
        self._model_name = (self._api_config.served_model_name or llm.model_id
                            or os.path.basename(model_dir) or model_dir
                            or "model")
        self._admission = _AdmissionController(
            self._api_config.max_queued_requests,
            self._api_config.queue_timeout,
        )
        self._capabilities = _capabilities_for(llm)
        self._close_lock = asyncio.Lock()
        self._close_task = None
        self._closed = False

    @property
    def llm(self) -> Union[LLM, TTS]:
        return self._llm

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def capabilities(self) -> EngineCapabilities:
        return self._capabilities

    @property
    def active_requests(self) -> int:
        return self._admission.active

    @property
    def queued_requests(self) -> int:
        return self._admission.waiting

    async def close(self) -> None:
        """Drain request ownership, then release the model runtime once."""
        async with self._close_lock:
            if self._closed:
                return
            if self._close_task is None:
                self._close_task = asyncio.create_task(self._close_runtime())
            close_task = self._close_task
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            # Shutdown cancellation must not destroy a runtime while native
            # work still owns it.
            await asyncio.shield(close_task)
            raise

    async def _close_runtime(self) -> None:
        await self._admission.close()
        try:
            await asyncio.to_thread(self._llm.close)
        finally:
            self._closed = True

    async def count_prompt_tokens(
        self,
        messages: List[Dict[str, Any]],
        *,
        tool_config: ToolConfig,
        enable_thinking: bool,
    ) -> int:
        prepared = await self.prepare_request(
            messages,
            SamplingParams(max_tokens=1, enable_thinking=enable_thinking),
            tools=tool_config.tools,
            tool_choice=tool_config.tool_choice,
            tool_config=tool_config,
        )
        try:
            count = await _run_sync(
                partial(self._llm._count_prepared_prompt_tokens,
                        prepared.request))
            if count is None:
                raise UnsupportedFeatureError(
                    "exact token counting is not available for media inputs")
            return count
        finally:
            prepared.release()

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: SamplingParams,
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_config: Optional[ToolConfig] = None,
        tool_parser: str = "auto",
        reasoning_parser: str = "none",
        prepared: Optional[PreparedRequest] = None,
    ) -> CompletionOutput:
        owned = prepared
        try:
            tool_config = tool_config or validate_tool_request(
                messages, tools, tool_choice)
            if owned is None:
                owned = await self.prepare_request(
                    messages,
                    sampling_params,
                    tools=tool_config.tools,
                    tool_choice=tool_config.tool_choice,
                    tool_config=tool_config,
                )
            operation = partial(
                self._llm._complete_prepared_request,
                owned.request,
                sampling_params,
                tool_config,
                tool_parser=tool_parser,
                reasoning_parser=reasoning_parser,
            )
            return await _run_sync(operation)
        except (ServerError, KeyError, TypeError, ValueError):
            raise
        except Exception as exc:
            raise EngineError(str(exc)) from exc
        finally:
            if owned is not None:
                owned.release()

    async def prepare_request(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: SamplingParams,
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_config: Optional[ToolConfig] = None,
    ) -> PreparedRequest:
        lease = await self._admission.reserve()
        try:
            request = await _run_sync(
                partial(
                    self._llm._make_generation_request,
                    messages,
                    sampling_params,
                    tools=tools,
                    tool_choice=tool_choice,
                    tool_config=tool_config,
                ))
            return PreparedRequest(request=request, lease=lease)
        except BaseException:
            lease.release()
            raise

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: SamplingParams,
        *,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        prepared: Optional[PreparedRequest] = None,
    ) -> AsyncGenerator[StreamDelta, None]:
        iterator = None
        owned = prepared or await self.prepare_request(
            messages,
            sampling_params,
            tools=tools,
            tool_choice=tool_choice,
        )
        try:
            iterator = self._llm.generate_stream(
                messages,
                sampling_params,
                tools=tools,
                tool_choice=tool_choice,
                prebuilt_request=owned.request,
            )
            async for item in _iterate_sync(iterator):
                yield item
        except (ServerError, KeyError, TypeError, ValueError):
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise EngineError(str(exc)) from exc
        finally:
            if iterator is not None:
                await asyncio.to_thread(_close_stream, iterator)
            owned.release()

    async def stream_with_audio(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: SamplingParams,
        audio_params: AudioParams,
        *,
        prepared: Optional[PreparedRequest] = None,
    ) -> AsyncGenerator[StreamDelta, None]:
        iterator = None
        owned = prepared or await self.prepare_request(messages,
                                                       sampling_params)
        try:
            iterator = self._llm.generate_stream_with_audio(
                messages,
                sampling_params,
                audio_params=audio_params,
                prebuilt_request=owned.request,
            )
            async for item in _iterate_sync(iterator):
                yield item
        except (asyncio.CancelledError, ServerError):
            raise
        except Exception as exc:
            raise EngineError(str(exc)) from exc
        finally:
            if iterator is not None:
                await asyncio.to_thread(_close_stream, iterator)
            owned.release()

    async def stream_speech(
        self,
        text: str,
        audio_params: AudioParams,
    ) -> AsyncGenerator[bytes, None]:
        lease = await self._admission.reserve()
        iterator = None
        try:
            iterator = self._llm.generate_speech_stream(text, audio_params)
            async for item in _iterate_sync(iterator):
                if item.audio_bytes:
                    yield item.audio_bytes
        except (asyncio.CancelledError, ServerError):
            raise
        except Exception as exc:
            raise EngineError(str(exc)) from exc
        finally:
            if iterator is not None:
                await asyncio.to_thread(_close_stream, iterator)
            lease.release()
