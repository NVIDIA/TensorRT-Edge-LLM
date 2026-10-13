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
import threading
from concurrent.futures import ThreadPoolExecutor
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
    in_flight_batching: bool = False


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


class _IFBAdmissionGate:
    """Rate limiter for an engine that schedules its own requests.

    The blocking runtime has a single generation slot, so _AdmissionController
    serializes: one active caller, a bounded queue of waiters. The request
    engine queues and overlaps requests itself, so this gate keeps only the
    bound — at most ``max_concurrency + max_queued`` requests holding leases
    at once, where the concurrency is the engine's batch capacity — and
    never parks a caller: a full gate is an immediate overload response, the
    HTTP translation of the engine's own submit() refusal. The lease guards
    total in-flight capacity, not exclusive access; sizing it from the queue
    depth alone would serialize the very batching the engine exists for
    whenever the queue is configured small.
    """

    def __init__(self,
                 max_queued_requests: int,
                 max_concurrency: int = 1) -> None:
        self._limit = max_queued_requests + max(1, max_concurrency)
        self._active = 0
        self._closing = False
        self._drained = asyncio.Event()
        self._drained.set()

    @property
    def active(self) -> int:
        return self._active

    @property
    def waiting(self) -> int:
        # Nothing ever waits here; a caller either holds a lease or was
        # refused. The property exists for interface parity with the
        # serializing controller.
        return 0

    async def reserve(self) -> "_AdmissionLease":
        if self._closing:
            raise ServerUnavailableError()
        if self._active >= self._limit:
            raise ServerOverloadedError()
        self._active += 1
        self._drained.clear()
        return _AdmissionLease(self)

    def release(self) -> None:
        self._active -= 1
        if self._active == 0:
            self._drained.set()

    async def close(self) -> None:
        """Reject new work and wait until every lease has been released."""
        self._closing = True
        await self._drained.wait()


class _HandleLatch:
    """Joins a request's engine handle with a cancellation that may arrive
    before it exists.

    The handle is published from the worker thread once ``submit()`` returns;
    the cancellation comes from the event loop when the client disconnects.
    Whichever happens second does the cancelling, so a disconnect that lands
    while the worker is still inside ``submit()`` still stops the request
    instead of leaving it to decode to ``max_tokens`` in its batch seat.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._handle = None
        self._cancelled = False

    def publish(self, handle) -> None:
        with self._lock:
            self._handle = handle
            if self._cancelled:
                handle.cancel()

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            if self._handle is not None:
                self._handle.cancel()


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
    speculative_method = str(config.get("spec_decode_type", "none"))
    if speculative_method == "gemma4_mtp":
        speculative_method = "mtp"

    chat = llm.runtime_kind == "chat"
    input_modalities = ["text"]
    if chat and layout.visual_dir:
        input_modalities.append("image")
        if llm.video_capable:
            input_modalities.append("video")
    if chat and layout.audio_dir:
        input_modalities.append("audio")
    output_modalities = ["text"] if chat else []
    # Under in-flight batching the engine serves text only; the Omni speech
    # stack is not loaded, so it must not be advertised either.
    speech = bool(layout.has_speech) and getattr(llm, "_engine", None) is None
    if speech:
        output_modalities.append("audio")

    max_kv = builder.get("max_kv_cache_capacity")
    max_positions = config.get("max_position_embeddings")
    max_model_len = max_kv if isinstance(max_kv, int) else max_positions
    # Sequences that can genuinely overlap: the engine's batch dimension under
    # in-flight batching, a single slot on the blocking path.
    in_flight = getattr(llm, "_engine", None) is not None
    builder_batch = builder.get("max_batch_size")
    max_num_seqs = 1
    if in_flight and isinstance(builder_batch, int) and builder_batch > 0:
        max_num_seqs = builder_batch
    return EngineCapabilities(
        chat=chat,
        transcription=chat and layout.has_transcription,
        speech=speech,
        input_modalities=tuple(input_modalities),
        output_modalities=tuple(output_modalities),
        max_model_len=max_model_len
        if isinstance(max_model_len, int) else None,
        max_input_len=builder.get("max_input_len") if isinstance(
            builder.get("max_input_len"), int) else None,
        max_batch_size=builder.get("max_batch_size") if isinstance(
            builder.get("max_batch_size"), int) else None,
        max_num_seqs=max_num_seqs,
        kv_cache_dtype=str(config.get("kv_cache_dtype", "unknown")),
        speculative_decoding=llm.has_draft_model,
        speculative_method=speculative_method,
        context_reuse=bool(getattr(llm, "context_cache_enabled", False)),
        in_flight_batching=in_flight,
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


async def _in_thread(executor, function, *args):
    """Run ``function`` off the event loop: on ``executor`` when the client has
    one of its own, otherwise on the loop's default pool."""
    if executor is None:
        return await asyncio.to_thread(function, *args)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, partial(function, *args))


async def _iterate_sync(iterator, executor=None):
    """Advance a native stream and cancel it before waiting on disconnect."""
    while True:
        next_task = asyncio.create_task(
            _in_thread(executor, _next_stream_item, iterator))
        try:
            item = await asyncio.shield(next_task)
        except asyncio.CancelledError:
            close_task = asyncio.create_task(
                _in_thread(executor, _close_stream, iterator))
            await asyncio.shield(
                asyncio.gather(next_task, close_task, return_exceptions=True))
            raise
        if item is _STREAM_END:
            return
        yield item


async def _run_sync(operation, on_cancel=None, executor=None):
    """Run blocking work without outliving the request that owns its lease.

    ``on_cancel`` runs first when the awaiting task is cancelled, before this
    waits for the worker: it is how a disconnected client's request is
    cancelled in the engine instead of being waited out to completion.
    """
    worker = asyncio.create_task(_in_thread(executor, operation))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        if on_cancel is not None:
            on_cancel()
        await asyncio.shield(asyncio.gather(worker, return_exceptions=True))
        raise


class EngineClient:
    """Asynchronous, bounded-queue adapter for one Edge-LLM runtime."""

    #: The dedicated pool for in-flight-batching streams; None means the
    #: loop's default pool (the blocking path, and clients built without
    #: __init__ in tests).
    _executor = None

    def __init__(self,
                 llm: Union[LLM, TTS],
                 api_config: Optional[ApiConfig] = None) -> None:
        self._llm = llm
        self._api_config = api_config or ApiConfig()
        model_dir = llm.model_dir
        self._model_name = (self._api_config.served_model_name or llm.model_id
                            or os.path.basename(model_dir) or model_dir
                            or "model")
        # The engine's own refusal, translated to the overload response where
        # the HTTP layer expects it. A tuple so the except clause below stays
        # valid when the native module (and thus the type) is absent.
        submit_error = getattr(getattr(llm, "_rt", None), "SubmitError", None)
        self._submit_errors = (submit_error, ) if submit_error else ()
        self._executor = None
        if getattr(llm, "_engine", None) is not None:
            self._admission = _IFBAdmissionGate(
                self._api_config.max_queued_requests,
                getattr(llm, "_max_batch_size", 1) or 1)
            # Every admitted stream parks one thread on its channel for its
            # whole life. The loop's default pool (min(32, cpus + 4)) is sized
            # for short hops, so under load the gate's worth of streams would
            # starve preparation, token counting and close() behind them:
            # give the streams a pool sized for the gate, plus headroom for
            # those short hops.
            self._executor = ThreadPoolExecutor(
                max_workers=self._admission._limit + 4,
                thread_name_prefix="edgellm-ifb")
        else:
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
            if self._executor is not None:
                self._executor.shutdown(wait=False)

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
            latch = _HandleLatch()
            operation = partial(
                self._llm._complete_prepared_request,
                owned.request,
                sampling_params,
                tool_config,
                tool_parser=tool_parser,
                reasoning_parser=reasoning_parser,
                on_handle=latch.publish,
            )
            return await _run_sync(operation,
                                   on_cancel=latch.cancel,
                                   executor=self._executor)
        except (ServerError, KeyError, TypeError, ValueError):
            raise
        except getattr(self, "_submit_errors", ()) as exc:
            raise ServerOverloadedError(str(exc)) from exc
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
        apply_chat_template: bool = True,
        add_generation_prompt: bool = True,
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
                    apply_chat_template=apply_chat_template,
                    add_generation_prompt=add_generation_prompt,
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
            async for item in _iterate_sync(iterator, self._executor):
                yield item
        except (ServerError, KeyError, TypeError, ValueError):
            raise
        except asyncio.CancelledError:
            raise
        except getattr(self, "_submit_errors", ()) as exc:
            raise ServerOverloadedError(str(exc)) from exc
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
