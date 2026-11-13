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
"""Qwen3-TTS runtime configuration artifacts."""

from typing import Any, Dict

from ...core import contracts


def _talker(root: Dict[str, Any]) -> Dict[str, Any]:
    return root.get("talker_config") or root


def update_talker_config(config: Dict[str, Any], root: Dict[str, Any], cfg,
                         args) -> None:
    """Patch runtime config for the Qwen3-TTS talker component."""
    talker = _talker(root)
    config["model"] = str(talker.get("model_type", "qwen3_tts_talker"))
    for key in ("tts_pad_token_id", "tts_bos_token_id", "tts_eos_token_id",
                "codec_nothink_id", "codec_think_bos_id", "codec_think_eos_id",
                "codec_pad_id", "codec_bos_id", "codec_eos_token_id",
                "codec_think_id", "accept_hidden_layer", "num_code_groups",
                "tts_model_type", "codec_language_id", "spk_is_dialect"):
        if key in talker:
            config[key] = talker[key]
        elif key in root:
            config[key] = root[key]
    thinker = root.get("thinker_config") or {}
    thinker_text = thinker.get("text_config") or {}
    config["thinker_hidden_size"] = int(
        talker.get(
            "thinker_hidden_size",
            talker.get("text_hidden_size", thinker_text.get("hidden_size",
                                                            0))))
    text_vocab = talker.get("text_vocab_size", thinker_text.get("vocab_size"))
    if text_vocab is not None:
        config["text_vocab_size"] = int(text_vocab)
    speaker = talker.get("speaker_id", talker.get("spk_id"))
    if isinstance(speaker, dict) and speaker:
        config["speaker_id"] = speaker
        config["default_speaker_id"] = talker.get("default_speaker_id",
                                                  next(iter(speaker.values())))
    config["num_deepstack_features"] = 0
    _ = cfg, args


def update_code_predictor_config(config: Dict[str, Any], root: Dict[str, Any],
                                 cfg, args) -> None:
    """Patch runtime config for the Qwen3-TTS code predictor."""
    talker = _talker(root)
    config["model"] = "qwen3_tts_code_predictor"
    config["use_embeddings_input"] = True
    config["num_code_groups"] = int(talker.get("num_code_groups", 16))
    config["num_deepstack_features"] = 0
    _ = cfg, args


def component_runtime_config(bundle, component: contracts.Component, args):
    """Return Qwen3-TTS component runtime config."""
    if component == contracts.Component.SPEAKER_ENCODER:
        speaker = dict(bundle.component_dict(component))
        return {
            "model_type": "qwen3_tts_speaker_encoder",
            "sample_rate": int(speaker.get("sample_rate", 24000)),
            "speaker_encoder_config": speaker,
            "tensor_contract": {
                "inputs": {
                    "wav": [1, "samples"],
                },
                "outputs": {
                    "speaker_embedding": [
                        1,
                        int(speaker.get("enc_dim", 1024)),
                    ],
                },
            },
            "builder_config": {
                "max_reference_samples": 960000,
            },
        }
    if component == contracts.Component.SPEECH_TOKENIZER_ENCODER:
        speech = bundle.root["_speech_tokenizer_config"]
        return {
            "model_type": "qwen3_tts_speech_tokenizer_encoder",
            "sample_rate": int(speech["input_sample_rate"]),
            "bucket_samples": int(speech["input_sample_rate"]) * 40,
            "num_quantizers": int(speech["encoder_valid_num_quantizers"]),
            "tensor_contract": {
                "inputs": {
                    "wav": [1, int(speech["input_sample_rate"]) * 40],
                },
                "outputs": {
                    "codes": [
                        int(speech["input_sample_rate"]) * 40 //
                        int(speech["encode_downsample_rate"]),
                        int(speech["encoder_valid_num_quantizers"]),
                    ],
                },
            },
        }
    if component != contracts.Component.CODE2WAV:
        return None
    code2wav = dict(bundle.component_dict(component))
    return {
        "model_type": "qwen3_tts_code2wav",
        "code2wav_config": code2wav,
        "builder_config": {
            "min_code_len": args.min_code_len,
            "opt_code_len": args.opt_code_len,
            "max_code_len": args.max_code_len,
        },
    }
