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
"""File contracts for one checkpoint-direct engine bundle."""

import enum
import json
import os
from dataclasses import dataclass
from typing import Optional


class EngineType(enum.Enum):
    LLM = "llm"
    SPEC_DECODE = "spec_decode"
    UNKNOWN = "unknown"


ONNX_MODEL_FILE = "model.onnx"
LLM_ENGINE_FILE = "llm.engine"
DLLM_ENGINE_FILE = "dllm.engine"
VISUAL_ENGINE_FILE = "visual.engine"
AUDIO_ENGINE_FILE = "audio_encoder.engine"
CODE2WAV_ENGINE_FILE = "code2wav.engine"
SPEC_BASE_ENGINE_FILE = "spec_base.engine"
SPEC_DRAFT_ENGINE_FILE = "spec_draft.engine"


@dataclass(frozen=True)
class BundleLayout:
    """Model components discovered in one checkpoint-direct bundle."""

    root: str
    engine_type: EngineType
    visual_dir: Optional[str] = None
    audio_dir: Optional[str] = None
    talker_dir: Optional[str] = None
    code_predictor_dir: Optional[str] = None
    code2wav_dir: Optional[str] = None
    audio_model_type: str = ""

    @property
    def media_dir(self) -> str:
        """Bundle root expected by the composable multimodal runtime."""
        return self.root if self.visual_dir or self.audio_dir else ""

    @property
    def has_speech(self) -> bool:
        return all(
            (self.talker_dir, self.code_predictor_dir, self.code2wav_dir))

    @property
    def has_transcription(self) -> bool:
        return bool(self.audio_dir and "asr" in self.audio_model_type.lower())


def is_onnx_dir(path: str) -> bool:
    """Return whether ``path`` is an ONNX export directory."""
    return os.path.isfile(os.path.join(path, ONNX_MODEL_FILE))


def validate_llm_engine_dir(engine_dir: str) -> bool:
    return any(
        os.path.isfile(os.path.join(engine_dir, filename))
        for filename in (LLM_ENGINE_FILE, DLLM_ENGINE_FILE))


def validate_spec_decode_engine_dir(engine_dir: str) -> bool:
    return all(
        os.path.isfile(os.path.join(engine_dir, name))
        for name in (SPEC_BASE_ENGINE_FILE, SPEC_DRAFT_ENGINE_FILE))


def detect_engine_type(engine_dir: str) -> EngineType:
    if validate_spec_decode_engine_dir(engine_dir):
        return EngineType.SPEC_DECODE
    if validate_llm_engine_dir(engine_dir):
        return EngineType.LLM
    return EngineType.UNKNOWN


def classify_model_source(path: str) -> str:
    """Classify a CLI model argument without inspecting checkpoint metadata."""
    if os.path.isdir(path):
        if detect_engine_type(path) != EngineType.UNKNOWN:
            return "engine_dir"
        if is_onnx_dir(path):
            return "onnx_dir"
    return "model"


def _component_dir(root: str, name: str, engine_file: str) -> Optional[str]:
    path = os.path.join(root, name)
    return path if os.path.isfile(os.path.join(path, engine_file)) else None


def _model_type(component_dir: Optional[str]) -> str:
    if component_dir is None:
        return ""
    try:
        with open(os.path.join(component_dir, "config.json"),
                  encoding="utf-8") as file:
            config = json.load(file)
        return str(config.get("model_type", ""))
    except (OSError, ValueError):
        return ""


def inspect_bundle(bundle_dir: str) -> BundleLayout:
    """Inspect a bundle once and return its immutable component contract."""
    root = os.path.abspath(bundle_dir)
    visual_dir = _component_dir(root, "visual", VISUAL_ENGINE_FILE)
    audio_dir = _component_dir(root, "audio", AUDIO_ENGINE_FILE)
    return BundleLayout(
        root=root,
        engine_type=detect_engine_type(root),
        visual_dir=visual_dir,
        audio_dir=audio_dir,
        talker_dir=_component_dir(root, "talker", LLM_ENGINE_FILE),
        code_predictor_dir=_component_dir(root, "code_predictor",
                                          LLM_ENGINE_FILE),
        code2wav_dir=_component_dir(root, "code2wav", CODE2WAV_ENGINE_FILE),
        audio_model_type=_model_type(audio_dir),
    )
