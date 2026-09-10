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
"""
Server-side audio content resolution.

The HTTP server accepts two OpenAI-compatible audio content forms in chat
messages:

  - ``{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}``
  - ``{"type": "audio_url", "audio_url": {"url": "https://... | file:///abs/path | data:audio/...;base64,..."}}``

HTTP(S) downloads and decoded payloads are bounded to the upload limit. Local
paths and ``file://`` are only accepted over HTTP when the server runs with
``--allowed-local-media-path``, and then only inside it.

This module resolves each accepted form to raw audio bytes and hands them
to the C++ runtime via ``_edgellm_runtime.load_audio_buffer_from_bytes``.
Container decode + resample run in C++ (miniaudio); the audio runner
auto-selects its mel feature extractor (``whisper`` for Qwen3-Omni /
Qwen3-ASR, ``parakeet`` for Nemotron-Omni) from
``<engine>/audio/config.json::model_type`` at init. The Python side
never touches a feature extractor or PCM samples.
"""

import logging
from typing import Any, Dict, List

from .media_source import (decode_base64_data_url, decode_base64_payload,
                           fetch_remote_media, resolve_file_url)

logger = logging.getLogger("edgellm.audio")

# Audio bytes are buffered in memory (and copied again as base64), and
# compressed audio expands further when decoded (the C++ loader also caps the
# decoded duration); 25 MiB matches the OpenAI upload limit, shared by the chat
# and transcription paths.
MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024


def _decode_data_url(url: str) -> bytes:
    """Decode ``data:audio/...;base64,<payload>`` URLs (lenient base64, the
    historical audio behavior)."""
    return decode_base64_data_url(url,
                                  "audio",
                                  strict=False,
                                  max_bytes=MAX_AUDIO_UPLOAD_BYTES)


def resolve_audio_message(item: Dict[str, Any]):
    """Resolve one audio content item to its underlying source.

    Returns one of:
      * ``str`` — a local filesystem path (caller will open it)
      * ``bytes`` — already-decoded audio bytes (wav/mp3/flac container)

    Raises ``ValueError`` for malformed, oversized, or unsupported sources.
    """
    content_type = item.get("type")

    if content_type == "input_audio":
        # OpenAI canonical form. `format` is not validated here: the C++ decoder
        # raises a clear error on an unsupported container.
        payload = item.get("input_audio") or {}
        data_b64 = payload.get("data")
        if not data_b64 or not isinstance(data_b64, str):
            raise ValueError("input_audio.data is required (base64 string)")
        return decode_base64_payload(data_b64,
                                     "input_audio.data",
                                     strict=False,
                                     max_bytes=MAX_AUDIO_UPLOAD_BYTES)

    if content_type == "audio_url":
        url = ((item.get("audio_url") or {}).get("url") or "").strip()
        if not url:
            raise ValueError("audio_url.url is required")
        if url.startswith("http://") or url.startswith("https://"):
            return fetch_remote_media(url, "audio", MAX_AUDIO_UPLOAD_BYTES)
        if url.startswith("data:"):
            return _decode_data_url(url)
        if url.startswith("file:"):
            return resolve_file_url(url)
        # Bare paths remain valid audio_url values and are confined by the
        # configured local-media root at load time.
        return url

    raise ValueError(f"Unsupported audio content type: {content_type!r}")


def load_audio_buffers(
    rt_module,
    messages: List[Dict[str, Any]],
) -> list:
    """Walk messages and return a list of ``rt.AudioData`` for the runtime.

    Mirrors ``engine._load_image_buffers``: hand raw encoded bytes to the
    C++ runtime via ``load_audio_buffer_from_bytes``; container decode runs
    in C++ (miniaudio) and the audio runner extracts mel internally per its
    ``audio/config.json``. Python touches neither PCM samples nor mel.
    """
    audios = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") not in ("input_audio", "audio_url"):
                continue
            source = resolve_audio_message(item)
            # ``resolve_audio_message`` returns either a path (str) or already
            # -decoded container bytes; normalise to bytes here so the runtime
            # API surface is single-typed.
            if isinstance(source, (bytes, bytearray)):
                audio_bytes = bytes(source)
            else:
                # Bounded read: one extra byte detects oversize without
                # buffering an unbounded file.
                with open(source, "rb") as f:
                    audio_bytes = f.read(MAX_AUDIO_UPLOAD_BYTES + 1)
                if len(audio_bytes) > MAX_AUDIO_UPLOAD_BYTES:
                    raise ValueError(
                        f"audio file exceeds the supported maximum "
                        f"of {MAX_AUDIO_UPLOAD_BYTES} bytes: {source}")
            audios.append(rt_module.load_audio_buffer_from_bytes(audio_bytes))
    return audios
