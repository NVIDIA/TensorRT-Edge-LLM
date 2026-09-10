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
"""Typed OpenAI protocol models supported by the Edge-LLM server."""

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import (BaseModel, ConfigDict, Field, StrictBool,
                      field_validator, model_validator)


class OpenAIBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StreamOptions(OpenAIBaseModel):
    include_usage: StrictBool = False


class AudioGenerationConfig(OpenAIBaseModel):
    voice: str = ""
    talker_temperature: float = Field(default=0.9, ge=0.0)
    talker_top_k: int = Field(default=50, ge=0)
    talker_top_p: float = Field(default=1.0, ge=0.0)
    repetition_penalty: float = Field(default=1.05, ge=0.0)
    max_audio_length: int = Field(default=4096, ge=1)
    codec_chunk_frames: int = Field(default=10, ge=1)
    talker_prefill_threshold: int = Field(default=4, ge=1)

    def generation_kwargs(self) -> Dict[str, Any]:
        return self.model_dump(include=set(AudioGenerationConfig.model_fields))


class ChatAudioConfig(AudioGenerationConfig):
    format: Literal["pcm16"] = "pcm16"


class ChatCompletionRequest(OpenAIBaseModel):
    messages: List[Dict[str, Any]]
    model: Optional[str] = None
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: bool = False
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=50)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    n: int = Field(default=1, ge=1, le=1)
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: bool = True
    response_format: Optional[Dict[str, Any]] = None
    modalities: Optional[List[Literal["text", "audio"]]] = None
    audio: Optional[ChatAudioConfig] = None
    enable_thinking: bool = False
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    disable_spec_decode: bool = False
    reuse_context: StrictBool = True
    cache_generated_tokens: StrictBool = True

    @field_validator("stop")
    @classmethod
    def _validate_stop(cls, value):
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, list) and all(
                isinstance(item, str) for item in value):
            return value
        raise ValueError("stop must be a string or an array of strings")

    @model_validator(mode="after")
    def _validate_request(self):
        if not self.messages:
            raise ValueError("messages must not be empty")
        if (self.max_tokens is not None
                and self.max_completion_tokens is not None
                and self.max_tokens != self.max_completion_tokens):
            raise ValueError(
                "max_tokens and max_completion_tokens must match when both "
                "are provided")
        if self.top_logprobs is not None and not self.logprobs:
            raise ValueError("top_logprobs requires logprobs=true")
        if self.chat_template_kwargs is not None:
            extra = set(self.chat_template_kwargs) - {"enable_thinking"}
            if extra:
                raise ValueError("unsupported chat_template_kwargs: " +
                                 ", ".join(sorted(extra)))
            value = self.chat_template_kwargs.get("enable_thinking")
            if value is not None and not isinstance(value, bool):
                raise ValueError(
                    "chat_template_kwargs.enable_thinking must be a bool")
            if value is not None:
                self.enable_thinking = value
        wants_audio = bool(self.modalities and "audio" in self.modalities)
        if wants_audio != (self.audio is not None):
            raise ValueError(
                "audio output requires both modalities including 'audio' "
                "and an audio configuration")
        return self

    @property
    def effective_max_tokens(self) -> int:
        return self.max_completion_tokens or self.max_tokens or 2048

    @property
    def stop_strings(self) -> List[str]:
        if self.stop is None:
            return []
        return [self.stop] if isinstance(self.stop, str) else self.stop


class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionMessage(OpenAIBaseModel):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    audio: Optional[Dict[str, Any]] = None


class ChatCompletionChoice(OpenAIBaseModel):
    index: int = 0
    message: ChatCompletionMessage
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


class ChatCompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class DeltaMessage(OpenAIBaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    audio: Optional[Dict[str, Any]] = None


class ChatCompletionStreamChoice(OpenAIBaseModel):
    index: int = 0
    delta: DeltaMessage
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(OpenAIBaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]
    usage: Optional[UsageInfo] = None


class ModelCard(OpenAIBaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "tensorrt-edgellm"
    max_model_len: Optional[int] = None


class ModelList(OpenAIBaseModel):
    object: Literal["list"] = "list"
    data: List[ModelCard]


class TranscriptionResponse(OpenAIBaseModel):
    text: str
    language: Optional[str] = None


class SpeechRequest(AudioGenerationConfig):
    model: Optional[str] = None
    input: str = Field(min_length=1)
    response_format: Literal["pcm", "pcm16"] = "pcm"
    speed: float = Field(default=1.0, gt=0.0, le=4.0)
