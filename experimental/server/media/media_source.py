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
"""Shared OpenAI media-source resolution and request inspection."""
from __future__ import annotations

import base64
import binascii
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Set
from urllib.parse import unquote, urlparse

REMOTE_MEDIA_TIMEOUT_SECONDS = 10


def decode_base64_payload(payload: str,
                          kind: str,
                          *,
                          strict: bool = False,
                          max_bytes: Optional[int] = None) -> bytes:
    """Decode a raw base64 payload to bytes. ``max_bytes`` is enforced exactly
    on the decoded size; an encoded-length pre-check rejects clearly oversized
    payloads first (sized so base64 padding never mis-rejects)."""
    if max_bytes is not None and len(payload) > -(-max_bytes // 3) * 4:
        raise ValueError(f"{kind} payload exceeds the supported "
                         f"maximum of {max_bytes} bytes")
    try:
        data = base64.b64decode(payload, validate=strict)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"invalid base64 {kind} payload: {exc}") from exc
    if max_bytes is not None and len(data) > max_bytes:
        raise ValueError(f"{kind} payload exceeds the supported "
                         f"maximum of {max_bytes} bytes")
    return data


def decode_base64_data_url(url: str,
                           kind: str,
                           *,
                           strict: bool = False,
                           max_bytes: Optional[int] = None) -> bytes:
    """Decode a ``data:<kind>/...;base64,<payload>`` URL to bytes (size policy
    per :func:`decode_base64_payload`)."""
    head, _, payload = url.partition(",")
    if ";base64" not in head:
        raise ValueError(f"data: {kind} URLs must use base64 encoding")
    return decode_base64_payload(payload,
                                 f"data: {kind}",
                                 strict=strict,
                                 max_bytes=max_bytes)


def resolve_file_url(url: str) -> str:
    """Convert a local ``file://`` URL to a filesystem path; non-local hosts
    and empty paths are client errors."""
    parsed = urlparse(url)
    if parsed.netloc and parsed.netloc not in ("", "localhost"):
        raise ValueError(
            f"file:// URL must be local (no host); got netloc={parsed.netloc!r}"
        )
    path = unquote(parsed.path)
    if not path:
        raise ValueError("file:// URL has empty path")
    return path


def _validate_remote_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        raise ValueError("remote media URL must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("remote media URL must not contain credentials")


def fetch_remote_media(url: str, kind: str, max_bytes: int) -> bytes:
    """Fetch one HTTP(S) source with bounded time and memory usage."""
    _validate_remote_url(url)
    request = urllib.request.Request(
        url, headers={"User-Agent": "TensorRT-Edge-LLM/experimental-server"})
    try:
        with urllib.request.urlopen(  # noqa: S310 - URL is validated above
                request,
                timeout=REMOTE_MEDIA_TIMEOUT_SECONDS) as response:
            _validate_remote_url(response.geturl())
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError:
                    declared_size = -1
                if declared_size > max_bytes:
                    raise ValueError(
                        f"remote {kind} exceeds the supported maximum of "
                        f"{max_bytes} bytes")
            data = response.read(max_bytes + 1)
    except ValueError:
        raise
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        raise ValueError(f"failed to fetch remote {kind}: {exc}") from exc
    if len(data) > max_bytes:
        raise ValueError(f"remote {kind} exceeds the supported maximum of "
                         f"{max_bytes} bytes")
    return data


#: Content-item keys carrying a media reference. Which spelling each loader
#: accepts differs (only ``video`` takes both a bare string and ``{"url": ...}``),
#: so the policy reads every key both ways rather than tracking the difference.
MEDIA_REF_KEYS = ("image", "video", "audio", "image_url", "video_url",
                  "audio_url")
MAX_IMAGE_SOURCE_BYTES = 32 * 1024 * 1024
_MEDIA_MODALITIES = {
    "image": "image",
    "image_url": "image",
    "video": "video",
    "video_url": "video",
    "audio": "audio",
    "input_audio": "audio",
    "audio_url": "audio",
}


def message_media_modalities(messages) -> Set[str]:
    """Return media modalities requested by OpenAI-style message blocks."""
    modalities: Set[str] = set()
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict):
                modality = _MEDIA_MODALITIES.get(item.get("type"))
                if modality:
                    modalities.add(modality)
    return modalities


def iter_item_media_refs(item: dict):
    """Yield every media URL/path in one content item, in either spelling."""
    if not isinstance(item, dict):
        return
    for key in MEDIA_REF_KEYS:
        ref = item.get(key)
        url = ref.get("url") if isinstance(ref, dict) else ref
        if isinstance(url, str):
            yield url
    # {"type": "video", "frames": [path, ...]} feeds video_sampling directly.
    for frame in item.get("frames") or []:
        if isinstance(frame, str):
            yield frame


def resolve_image_message(item: dict):
    """Resolve an OpenAI ``image_url`` or Edge-LLM ``image`` block."""
    key = "image_url" if item.get("type") == "image_url" else "image"
    ref = item.get(key)
    url = ref.get("url") if isinstance(ref, dict) else ref
    if not isinstance(url, str) or not url:
        raise ValueError(f"{key} must contain a non-empty URL or path")
    if url.startswith(("http://", "https://")):
        return fetch_remote_media(url, "image", MAX_IMAGE_SOURCE_BYTES)
    if url.startswith("data:"):
        return decode_base64_data_url(url,
                                      "image",
                                      strict=True,
                                      max_bytes=MAX_IMAGE_SOURCE_BYTES)
    if url.startswith("file:"):
        return resolve_file_url(url)
    return url


def enforce_local_media_policy(messages, allowed_root: str) -> None:
    """Permit server-local media only below an explicitly allowed root."""
    root = Path(allowed_root).resolve() if allowed_root else None
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            for ref in iter_item_media_refs(item):
                if ref.startswith(("data:", "http://", "https://")):
                    continue
                if root is None:
                    raise PermissionError(
                        "local media paths are disabled; use base64 data URLs "
                        "or launch with --allowed-local-media-path")
                path = resolve_file_url(ref) if ref.startswith(
                    "file:") else ref
                if not Path(path).resolve().is_relative_to(root):
                    raise PermissionError("local media path is outside "
                                          f"--allowed-local-media-path: {ref}")
