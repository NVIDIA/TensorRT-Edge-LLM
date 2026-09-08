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
"""Micro-batching of compatible non-streaming generation requests.

The C++ runtime processes one ``LLMGenerationRequest`` at a time, but that
request may carry several rows (``requests``) that decode together up to the
engine's ``max_batch_size``. Decode is memory-bandwidth bound, so rows in one
call cost little more than a single row. The batcher collects concurrent
HTTP requests whose generation-level settings match, merges their rows into
one runtime call, and splits the response back per caller.
"""

import logging
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger("edgellm.batching")


class BatcherOverflow(RuntimeError):
    """The pending queue is at capacity; the caller should shed load."""


# Rows may only share one runtime call when these request-level settings
# match. Per-row inputs (messages, media, stop strings, logit bias) live under
# ``LLMGenerationRequest.requests`` and are free to differ.
BATCH_COMPATIBILITY_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "max_generate_length",
    "lora_weights_name",
    "save_system_prompt_kv_cache",
    "apply_chat_template",
    "add_generation_prompt",
    "enable_thinking",
    "disable_spec_decode",
    "recurrent_capture_interval",
    "num_logprobs",
    "context_cache_lookup_policy",
    "context_cache_commit_policy",
    "context_cache_replay_tail_length",
)


@dataclass
class ResponseSlice:
    """The rows of one runtime response that belong to one submitted request.

    Exposes the same attribute names as ``LLMGenerationResponse`` so callers
    can consume either interchangeably.
    """

    output_texts: List[str]
    output_ids: List[List[int]]
    finish_reasons: List[Any]
    logprobs: List[Any]
    prompt_token_counts: List[int]


@dataclass
class _QueuedRequest:
    request: Any
    future: Future


def resolve_batch_size(engine_max_batch_size: int,
                       max_queue_batch_size: Optional[int]) -> int:
    """Effective micro-batch size: the requested cap, bounded by the engine."""
    if max_queue_batch_size is not None:
        if 0 < engine_max_batch_size < max_queue_batch_size:
            logger.warning(
                "Capping max_queue_batch_size=%d to engine max_batch_size=%d",
                max_queue_batch_size, engine_max_batch_size)
            return engine_max_batch_size
        return max_queue_batch_size
    return engine_max_batch_size or 1


def _batch_key(request) -> Tuple[Any, ...]:
    return tuple(
        getattr(request, field) for field in BATCH_COMPATIBILITY_FIELDS)


def _is_batchable(request, video_requires_singleton: bool) -> bool:
    """Video rows on runners that enqueue per-clip tubelet shapes run alone."""
    if not video_requires_singleton:
        return True
    try:
        for row in request.requests:
            for buf in row.image_buffers:
                if buf.is_video:
                    return False
    except Exception:
        # A request that cannot be introspected runs alone.
        return False
    return True


def _copy_batch_settings(source, target) -> None:
    for field in BATCH_COMPATIBILITY_FIELDS:
        setattr(target, field, getattr(source, field))
    target.stream_channels = []


def _slice_response(response, start: int, count: int) -> ResponseSlice:
    end = start + count
    logprobs = getattr(response, "logprobs", None) or []
    prompt_tokens = getattr(response, "prompt_token_counts", None) or []
    return ResponseSlice(
        output_texts=list(response.output_texts[start:end]),
        output_ids=[list(ids) for ids in response.output_ids[start:end]],
        finish_reasons=list(response.finish_reasons[start:end]),
        logprobs=list(logprobs[start:end]),
        prompt_token_counts=list(prompt_tokens[start:end]),
    )


class RequestBatcher:
    """Merge compatible requests and serialize runtime calls on one worker.

    ``runtime_handler`` receives a merged ``LLMGenerationRequest`` and returns
    the runtime response; it is expected to hold whatever lock the runtime
    needs. ``submit`` blocks the calling thread until its slice is available.
    """

    def __init__(
        self,
        runtime_handler: Callable[[Any], Any],
        max_batch_size: int,
        timeout_ms: float,
        max_pending: Optional[int] = None,
        video_requires_singleton: bool = False,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if timeout_ms < 0:
            raise ValueError("timeout_ms must be non-negative")
        if max_pending is not None and max_pending < 1:
            raise ValueError("max_pending must be positive")

        self._runtime_handler = runtime_handler
        self._max_batch_size = max_batch_size
        self._timeout_s = timeout_ms / 1000.0
        self._max_pending = max_pending
        self._video_requires_singleton = video_requires_singleton
        self._cv = threading.Condition()
        self._queue: List[_QueuedRequest] = []
        self._closed = False
        self._worker = threading.Thread(target=self._run,
                                        name="edgellm-request-batcher",
                                        daemon=True)
        self._worker.start()

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    @property
    def timeout_ms(self) -> float:
        return self._timeout_s * 1000.0

    @property
    def pending(self) -> int:
        with self._cv:
            return len(self._queue)

    def submit(self, request) -> ResponseSlice:
        future: Future = Future()
        with self._cv:
            if self._closed:
                raise RuntimeError("Request batcher is closed")
            if (self._max_pending is not None
                    and len(self._queue) >= self._max_pending):
                raise BatcherOverflow(
                    f"batcher queue full ({self._max_pending} pending)")
            self._queue.append(_QueuedRequest(request=request, future=future))
            self._cv.notify()
        return future.result()

    def close(self) -> None:
        """Stop accepting work; queued requests still complete."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._worker.join(timeout=5.0)

    def _run(self) -> None:
        # The worker must outlive any single failure: a dead worker would
        # leave every pending Future, and the caller thread parked on it,
        # blocked forever.
        while True:
            try:
                batch = self._take_batch()
            except Exception:
                logger.exception("Batch selection failed; continuing")
                continue
            if batch is None:
                return
            if batch:
                self._process_batch(batch)

    def _take_batch(self) -> Optional[List[_QueuedRequest]]:
        with self._cv:
            while not self._queue and not self._closed:
                self._cv.wait()
            if not self._queue:
                return None

            first = self._queue.pop(0)
            try:
                key = _batch_key(first.request)
            except Exception as exc:
                first.future.set_exception(exc)
                return []
            batch = [first]
            if not _is_batchable(first.request,
                                 self._video_requires_singleton):
                return batch
            deadline = time.monotonic() + self._timeout_s
            while len(batch) < self._max_batch_size:
                self._move_compatible_locked(batch, key)
                if len(batch) >= self._max_batch_size or self._closed:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cv.wait(remaining)
            return batch

    def _move_compatible_locked(self, batch: List[_QueuedRequest],
                                key: Tuple[Any, ...]) -> None:
        idx = 0
        while idx < len(self._queue) and len(batch) < self._max_batch_size:
            item = self._queue[idx]
            try:
                compatible = (_batch_key(
                    item.request) == key and _is_batchable(
                        item.request, self._video_requires_singleton))
            except Exception:
                # Unkeyable requests are popped first on a later round and
                # fail there on their own.
                compatible = False
            if compatible:
                batch.append(self._queue.pop(idx))
            else:
                idx += 1

    def _process_batch(self, batch: List[_QueuedRequest]) -> None:
        try:
            response = self._runtime_handler(self._merge(batch))
            offset = 0
            for item in batch:
                count = len(item.request.requests)
                item.future.set_result(_slice_response(response, offset,
                                                       count))
                offset += count
        except Exception as exc:
            logger.exception("Batched inference failed")
            for item in batch:
                item.future.set_exception(exc)

    @staticmethod
    def _merge(batch: List[_QueuedRequest]):
        if len(batch) == 1:
            return batch[0].request
        first = batch[0].request
        merged = type(first)()
        _copy_batch_settings(first, merged)
        rows = []
        for item in batch:
            rows.extend(item.request.requests)
        merged.requests = rows
        return merged
