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
"""Checkpoint-native inference server for TensorRT Edge-LLM.

Public API::

    from experimental.server import LLM, SamplingParams

    llm = LLM(model="Qwen/Qwen3.5-0.8B")
    outputs = llm.generate(["Hello!"], SamplingParams(max_tokens=64))
    print(outputs[0].text)

    # Or start an OpenAI-compatible server:
    llm.serve(port=8000)
"""

from .config import ContextCacheConfig
from .runtime.engine import (LLM, TTS, AudioParams, CompletionOutput,
                             SamplingParams, StreamDelta, load_model)
from .runtime.engine_build import clear_engine_cache, prune_engine_cache

__all__ = [
    "LLM", "TTS", "SamplingParams", "CompletionOutput", "StreamDelta",
    "AudioParams", "load_model", "ContextCacheConfig", "clear_engine_cache",
    "prune_engine_cache"
]
