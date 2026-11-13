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

import logging
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import (BaseModel, ConfigDict, Field, StrictBool, StrictInt,
                      ValidationInfo, field_validator, model_serializer,
                      model_validator)

logger = logging.getLogger("edgellm.server.protocol")

_UNSUPPORTED_SAMPLING_DEFAULTS = {
    "min_p": 0.0,
    "repetition_penalty": 1.0,
}


class OpenAIBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


Stop = Optional[Union[str, List[str]]]


def _validate_stop_value(value: Stop) -> Stop:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list) and all(
            isinstance(item, str) for item in value):
        return value
    raise ValueError("stop must be a string or an array of strings")


def _stop_strings(value: Stop) -> List[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else value


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
    seed: Optional[int] = Field(default=None, ge=0, le=(1 << 64) - 1)
    stop: Stop = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int = Field(default=50, ge=-1, le=1024)  # kMaxSamplingTopK
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: bool = True
    response_format: Optional[Dict[str, Any]] = None
    guided_decoding: Optional[Dict[str, Any]] = None
    modalities: Optional[List[Literal["text", "audio"]]] = None
    audio: Optional[ChatAudioConfig] = None
    apply_chat_template: StrictBool = True
    add_generation_prompt: StrictBool = True
    enable_thinking: bool = False
    reasoning_effort: Optional[str] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    disable_spec_decode: bool = False
    ignore_eos: bool = False
    reuse_context: StrictBool = True
    # None lets the deployment choose; see SamplingParams.cache_generated_tokens.
    cache_generated_tokens: Optional[StrictBool] = None
    priority: StrictInt = Field(default=0, ge=-(2**31), le=2**31 - 1)

    @field_validator("stop")
    @classmethod
    def _validate_stop(cls, value):
        return _validate_stop_value(value)

    @field_validator("min_p", "repetition_penalty")
    @classmethod
    def _normalize_unsupported_sampling_fields(
            cls, value: Union[int, float],
            info: ValidationInfo) -> Union[int, float]:
        default = _UNSUPPORTED_SAMPLING_DEFAULTS[info.field_name]
        if value == default:
            return value
        logger.warning("%s=%s is unsupported; using default %s",
                       info.field_name, value, default)
        return default

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
            extra = set(self.chat_template_kwargs) - {
                "enable_thinking", "reasoning_effort"
            }
            if extra:
                raise ValueError("unsupported chat_template_kwargs: " +
                                 ", ".join(sorted(extra)))
            value = self.chat_template_kwargs.get("enable_thinking")
            if value is not None and not isinstance(value, bool):
                raise ValueError(
                    "chat_template_kwargs.enable_thinking must be a bool")
            if value is not None:
                self.enable_thinking = value
            effort = self.chat_template_kwargs.get("reasoning_effort")
            if effort is not None and not isinstance(effort, str):
                raise ValueError(
                    "chat_template_kwargs.reasoning_effort must be a string")
            if effort is not None:
                self.reasoning_effort = effort
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
        return _stop_strings(self.stop)


class CompletionRequest(OpenAIBaseModel):
    prompt: Union[str, List[StrictInt]]
    model: Optional[str] = None
    max_tokens: int = Field(default=16, ge=1)
    n: Literal[1] = 1
    echo: Literal[False] = False
    logprobs: None = None
    stop: Stop = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0, le=1024)  # kMaxSamplingTopK
    disable_spec_decode: bool = False
    ignore_eos: bool = False
    reuse_context: StrictBool = True
    cache_generated_tokens: Optional[StrictBool] = None
    priority: StrictInt = Field(default=0, ge=-(2**31), le=2**31 - 1)

    @field_validator("prompt")
    @classmethod
    def _validate_prompt(cls, value):
        if isinstance(value, str):
            if not value:
                raise ValueError("prompt must not be empty")
            return value
        if not value:
            raise ValueError("prompt token IDs must not be empty")
        if any(token < 0 or token > 2**31 - 1 for token in value):
            raise ValueError(
                "prompt token IDs must be non-negative int32 values")
        return value

    @field_validator("stop")
    @classmethod
    def _validate_stop(cls, value):
        return _validate_stop_value(value)

    @property
    def stop_strings(self) -> List[str]:
        return _stop_strings(self.stop)

    @property
    def prompt_token_count(self) -> Optional[int]:
        return len(self.prompt) if isinstance(self.prompt, list) else None


class PromptTokensDetails(OpenAIBaseModel):
    cached_tokens: int = 0


class UsageInfo(OpenAIBaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: Optional[PromptTokensDetails] = None

    @model_serializer(mode="wrap")
    def _omit_unreported_details(self, handler):
        # Absent rather than null when the runtime did not report its reuse,
        # so a client cannot read a missing count as zero cached tokens.
        data = handler(self)
        if self.prompt_tokens_details is None:
            data.pop("prompt_tokens_details", None)
        return data


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


class CompletionChoice(OpenAIBaseModel):
    index: int = 0
    text: str
    logprobs: None = None
    finish_reason: Optional[str] = None


class CompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


class CompletionStreamResponse(OpenAIBaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
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
