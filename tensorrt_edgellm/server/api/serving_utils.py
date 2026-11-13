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
"""Shared OpenAI serving response helpers."""

import json
from typing import Any, AsyncGenerator, List, Optional

from ..runtime import engine
from . import errors, protocol


def usage(prompt_tokens: Optional[int],
          completion_tokens: int,
          cached_tokens: Optional[int] = None) -> protocol.UsageInfo:
    prompt_count = prompt_tokens or 0
    details = (protocol.PromptTokensDetails(
        cached_tokens=cached_tokens) if cached_tokens is not None else None)
    return protocol.UsageInfo(
        prompt_tokens=prompt_count,
        completion_tokens=completion_tokens,
        total_tokens=prompt_count + completion_tokens,
        prompt_tokens_details=details,
    )


def sse(model: Any) -> str:
    return "data: " + model.model_dump_json(exclude_none=False) + "\n\n"


def sse_error(error: errors.ServerError) -> str:
    return "data: " + json.dumps(error.payload()) + "\n\n"


async def await_first_output(
    deltas: AsyncGenerator[engine.StreamDelta, None]
) -> AsyncGenerator[engine.StreamDelta, None]:
    """Read ``deltas`` up to the first one carrying output, or to their end.

    A streaming handler calls this before its first chunk, so a request that
    fails before producing anything raises before the response starts.

    Returns:
        An iterator over all of ``deltas``, starting with the ones read here.
    """
    received: List[engine.StreamDelta] = []
    try:
        async for delta in deltas:
            received.append(delta)
            if delta.token_ids or delta.text or delta.audio_bytes:
                break
    except (KeyError, TypeError, ValueError) as exc:
        raise errors.InvalidRequestError(str(exc)) from exc
    return _replay(received, deltas)


async def _replay(received: List[engine.StreamDelta],
                  deltas: AsyncGenerator[engine.StreamDelta, None]):
    try:
        for delta in received:
            yield delta
        async for delta in deltas:
            yield delta
    finally:
        await deltas.aclose()


def check_cache_generated_tokens(sampling: engine.SamplingParams,
                                 llm: Any) -> None:
    """Reject a commit policy the deployment cannot publish with HTTP 400."""
    try:
        engine.resolve_cache_generated_tokens(
            sampling, getattr(llm, "publishes_prefill_state_only", False))
    except ValueError as exc:
        raise errors.InvalidRequestError(
            str(exc), param="cache_generated_tokens") from exc
