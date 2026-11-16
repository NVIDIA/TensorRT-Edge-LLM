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
"""Qwen3-TTS checkpoint weight mapping."""

import os

_PREFIXES = {
    "talker": ("talker.", ),
    "code-predictor": ("talker.code_predictor.", "code_predictor."),
    "code2wav": ("code2wav.", "speech_tokenizer.decoder.", "decoder."),
}


def checkpoint_dir(model_dir: str, component: str) -> str:
    """Select the speech-tokenizer checkpoint for Code2Wav."""
    speech_tokenizer = os.path.join(model_dir, "speech_tokenizer")
    if (component in ("code2wav", "speech-tokenizer-encoder")
            and os.path.isdir(speech_tokenizer) and any(
                name.endswith(".safetensors")
                for name in os.listdir(speech_tokenizer))):
        return speech_tokenizer
    return model_dir


def resolve_candidates(name: str, *, component: str, spec_type: str,
                       spec_role: str, quant_type: str):
    """Map frontend tensor names to Qwen3-TTS component checkpoints."""
    del spec_type, spec_role, quant_type
    candidates = [prefix + name for prefix in _PREFIXES.get(component, ())]
    if component == "talker" and name == "model.embed_tokens.weight":
        candidates.extend(
            ("talker.model.codec_embedding.weight",
             "talker.codec_embedding.weight", "codec_embedding.weight"))
    if component == "talker" and name.startswith("lm_head."):
        suffix = name[len("lm_head."):]
        candidates.extend(
            (f"talker.codec_head.{suffix}", f"codec_head.{suffix}"))
    return tuple(candidates)
