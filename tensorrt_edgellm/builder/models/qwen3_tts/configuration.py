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
"""Qwen3-TTS checkpoint and component configuration."""

import json
import logging
import os

from ...core import contracts

logger = logging.getLogger(__name__)


def prepare_root(model_dir: str, root: dict) -> dict:
    root = dict(root)
    speech_config = os.path.join(model_dir, "speech_tokenizer", "config.json")
    if os.path.isfile(speech_config):
        with open(speech_config) as config_file:
            root["_speech_tokenizer_config"] = json.load(config_file)
    return root


def available_components(root: dict, registered):
    """Expose voice-clone encoders only for Base checkpoints."""
    available = set(registered)
    has_speaker_encoder = isinstance(root.get("speaker_encoder_config"), dict)
    if not has_speaker_encoder:
        available.discard(contracts.Component.SPEAKER_ENCODER)
        available.discard(contracts.Component.SPEECH_TOKENIZER_ENCODER)
    speech = root.get("_speech_tokenizer_config") or {}
    if (has_speaker_encoder
            and not isinstance(speech.get("encoder_config"), dict)):
        available.discard(contracts.Component.SPEECH_TOKENIZER_ENCODER)
    return frozenset(available)


def setup_profiles(builder, builder_config, network, args, bundle) -> bool:
    """Install the raw-waveform profile consumed by CloneEncoderRunner."""
    if args.resolved_component != contracts.Component.SPEAKER_ENCODER:
        return False
    sample_rate = int(
        bundle.root.get("speaker_encoder_config",
                        {}).get("sample_rate", 24000))
    maximum = sample_rate * 40
    optimum = sample_rate * 10
    profile = builder.create_optimization_profile()
    profile.set_shape("wav", (1, 1024), (1, optimum), (1, maximum))
    builder_config.add_optimization_profile(profile)
    return True


def component_config(root: dict, component: contracts.Component) -> dict:
    root_type = str(root.get("model_type", ""))
    if component == contracts.Component.TALKER:
        talker = root.get("talker_config")
        if isinstance(talker, dict):
            return talker
        if root_type in ("qwen3_tts", "qwen3_tts_talker"):
            return root
        raise ValueError("Qwen3-TTS checkpoint has no talker_config")
    if component == contracts.Component.CODE_PREDICTOR:
        talker = root.get("talker_config") or root
        predictor = talker.get("code_predictor_config")
        if isinstance(predictor, dict):
            return predictor
        if root_type == "qwen3_tts_code_predictor":
            return root
        raise ValueError("Qwen3-TTS checkpoint has no code_predictor_config")
    if component == contracts.Component.CODE2WAV:
        talker = root.get("talker_config") or {}
        speech = root.get("_speech_tokenizer_config") or {}
        return (root.get("code2wav_config") or talker.get("code2wav_config")
                or speech.get("decoder_config") or root)
    if component == contracts.Component.SPEAKER_ENCODER:
        return dict(root["speaker_encoder_config"])
    if component == contracts.Component.SPEECH_TOKENIZER_ENCODER:
        return dict(root["_speech_tokenizer_config"]["encoder_config"])
    raise ValueError(f"Qwen3-TTS has no {component.value} configuration")


def multistage_topology(bundle, components, args) -> dict | None:
    """Text in, speech out: the talker speaks what it is given, with no backbone.

    The talker binds no `hidden` input here: the tts layout strips a fixed
    wrapper off the rendered turn and looks every row up in the text embedding
    table.
    """
    from ...core.artifacts.multistage import artifact_of

    present = frozenset(components)
    required = {
        contracts.Component.TALKER, contracts.Component.CODE_PREDICTOR,
        contracts.Component.CODE2WAV
    }
    if not required.issubset(present):
        logger.info("No multistage_config.json: the speech graph needs the "
                    "talker, code predictor and code2wav")
        return None

    talker = {
        "id":
        "talker",
        "runner":
        "qwen3_omni.talker",
        "artifact":
        artifact_of(contracts.Component.TALKER),
        "inputs": [
            {
                "name": "prompt_ids",
                "from": "request:messages",
                "on_missing": "fail"
            },
            {
                "name": "voice",
                "from": "request:voice",
                "on_missing": "skip"
            },
            {
                "name": "sampling",
                "from": "request:sampling",
                "on_missing": "skip"
            },
            {
                "name": "speaker_codes",
                "from": "request:speaker_codes",
                "on_missing": "skip"
            },
            {
                "name": "style",
                "from": "request:style",
                "on_missing": "skip"
            },
        ],
        "emits": [{
            "id": "codes",
            "kind": "codec",
            "transport": "chunk_stream"
        }],
        "config": {
            # Qwen3-TTS speaks one assistant turn with no backbone in front of it, which is
            # a different row plan from the segments layout the omni graph uses.
            "prefill_builder":
            "tts",
            "code_predictor_artifact":
            "../" + artifact_of(contracts.Component.CODE_PREDICTOR),
            # Verify window for code predictor speculation; 0 leaves the
            # autoregressive RVQ loop in place.
            "cp_spec_verify_size":
            0,
        },
    }
    if contracts.Component.SPEAKER_ENCODER in present:
        # Base checkpoints ship the reference encoders; CustomVoice ones do not, and the
        # talker only loads them when the graph names where they live. Offering the
        # waveform edge without them would admit a request the talker must then refuse.
        talker["inputs"].append({
            "name": "ref_audio",
            "from": "request:ref_audio",
            "on_missing": "skip"
        })
        talker["config"]["clone_encoder_artifact"] = "../" + artifact_of(
            contracts.Component.SPEAKER_ENCODER)
    code2wav = {
        "id":
        "code2wav",
        "runner":
        "qwen3_omni.code2wav",
        "artifact":
        artifact_of(contracts.Component.CODE2WAV),
        "inputs": [{
            "name": "codes",
            "from": "talker:codes",
            "on_missing": "fail"
        }],
        "emits": [{
            "id": "audio",
            "kind": "audio",
            "transport": "chunk_stream",
            "terminal": True
        }],
    }
    return {"terminals": ["code2wav:audio"], "stages": [talker, code2wav]}
