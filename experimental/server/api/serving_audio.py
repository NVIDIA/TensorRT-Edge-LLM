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
"""OpenAI transcription serving logic."""

import asyncio
import base64
import os
from typing import Optional

from ..media.audio_preprocess import MAX_AUDIO_UPLOAD_BYTES
from ..runtime.engine import AudioParams, SamplingParams
from ..runtime.engine_client import EngineClient
from .errors import (InvalidRequestError, ModelNotFoundError,
                     PayloadTooLargeError, UnsupportedFeatureError)
from .protocol import SpeechRequest, TranscriptionResponse

_QWEN3_ASR_LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "hu": "Hungarian",
    "mk": "Macedonian",
    "ro": "Romanian",
}
_QWEN3_ASR_LANGUAGE_NAMES.update({
    language.lower(): language
    for language in list(_QWEN3_ASR_LANGUAGE_NAMES.values())
})


def _asr_language_name(language: str) -> Optional[str]:
    """Return a provider-supported Qwen3-ASR language name.

    This is intentionally a model capability allowlist, not a general ISO
    language database. Accepting every language known to a locale package
    would advertise inputs that the loaded ASR model cannot honor.
    """
    if not language:
        return None
    return _QWEN3_ASR_LANGUAGE_NAMES.get(language.strip().lower())


class OpenAIServingTranscription:
    """OpenAI transcription handling for Qwen3-ASR-capable runtimes."""

    def __init__(self, engine_client: EngineClient) -> None:
        self._client = engine_client

    async def transcribe(
        self,
        file,
        *,
        model: str = "",
        prompt: str = "",
        language: str = "",
        temperature: float = 0.0,
    ) -> TranscriptionResponse:
        """Validate, preprocess, and transcribe one uploaded audio file."""
        if model and model != self._client.model_name:
            raise ModelNotFoundError(
                f"model {model!r} is not served by this process")
        if not 0.0 <= temperature <= 2.0:
            raise InvalidRequestError("temperature must be in [0, 2]",
                                      param="temperature")

        capabilities = self._client.capabilities
        if "audio" not in capabilities.input_modalities:
            raise UnsupportedFeatureError(
                "the loaded engine has no audio encoder",
                param="file",
            )
        if not capabilities.transcription:
            raise UnsupportedFeatureError(
                "the loaded model supports audio chat but not transcription",
                param="model",
            )
        language_name = _asr_language_name(language)
        if language and language_name is None:
            raise InvalidRequestError(f"unsupported language {language!r}",
                                      param="language")

        raw = await file.read(MAX_AUDIO_UPLOAD_BYTES + 1)
        if len(raw) > MAX_AUDIO_UPLOAD_BYTES:
            raise PayloadTooLargeError(
                "audio upload exceeds the 25 MiB server limit",
                param="file",
            )
        if not raw:
            raise InvalidRequestError("audio file is empty", param="file")

        extension = (os.path.splitext(file.filename
                                      or "")[1].lstrip(".").lower() or "wav")
        encoded = await asyncio.to_thread(
            lambda: base64.b64encode(raw).decode("ascii"))
        messages = []
        if language:
            messages.append({
                "role": "system",
                "content": language_name,
            })
        content = [{
            "type": "input_audio",
            "input_audio": {
                "data": encoded,
                "format": extension,
            },
        }]
        if prompt:
            content.append({"type": "text", "text": prompt})
        messages.append({
            "role": "user",
            "content": content,
        })
        output = await self._client.generate(
            messages,
            SamplingParams(
                temperature=temperature,
                top_p=1.0,
                top_k=1,
                max_tokens=4096,
            ),
        )

        text = output.text.replace("<|im_end|>", "").strip()
        detected: Optional[str] = None
        if "<asr_text>" in text:
            prefix, text = text.split("<asr_text>", 1)
            text = text.strip()
            for line in prefix.strip().splitlines():
                line = line.strip()
                if not line or line.lower() == "language none":
                    continue
                detected = (line[len("language "):].strip()
                            if line.lower().startswith("language ") else line)
                break
        return TranscriptionResponse(text=text, language=detected)


class OpenAIServingSpeech:
    """OpenAI speech generation backed by an Omni/TTS runtime."""

    def __init__(self, engine_client: EngineClient) -> None:
        self._client = engine_client

    def validate(self, request: SpeechRequest) -> AudioParams:
        """Validate speech controls against the loaded TTS runtime."""
        if request.model and request.model != self._client.model_name:
            raise ModelNotFoundError(
                f"model {request.model!r} is not served by this process")
        if not self._client.capabilities.speech:
            raise UnsupportedFeatureError(
                "the loaded model has no speech-generation components")
        if request.speed != 1.0:
            raise UnsupportedFeatureError(
                "speech speed control is not supported", param="speed")
        voices = self._client.llm.list_voices()
        if request.voice and voices and request.voice not in voices:
            raise InvalidRequestError(
                f"unknown voice {request.voice!r}; available: " +
                ", ".join(voices),
                param="voice",
            )
        return AudioParams(**request.generation_kwargs())

    async def stream(self, request: SpeechRequest):
        """Yield PCM chunks from a validated speech request."""
        params = self.validate(request)
        async for chunk in self._client.stream_speech(request.input, params):
            yield chunk
