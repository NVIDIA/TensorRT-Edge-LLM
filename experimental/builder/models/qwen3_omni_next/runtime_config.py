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
"""Qwen3-Omni-Next runtime configuration artifacts."""

from typing import Any, Dict

from ...core import contracts
from ...core.artifacts.runtime_config import normalize_rope_scaling


def _thinker(root: Dict[str, Any]) -> Dict[str, Any]:
    return root.get("thinker_config") or root


def _talker(root: Dict[str, Any]) -> Dict[str, Any]:
    return root.get("talker_config") or root


def update_llm_config(config: Dict[str, Any], root: Dict[str, Any], cfg,
                      args) -> None:
    """Identify the dense or sparse-MoE Next thinker to the runtime."""
    config["model"] = ("qwen3_omni_next_text_moe"
                       if cfg.num_experts > 0 else "qwen3_omni_next_text")
    config["num_deepstack_features"] = cfg.num_deepstack_features
    _ = root, args


def update_talker_config(config: Dict[str, Any], root: Dict[str, Any], cfg,
                         args) -> None:
    """Write the chunked Next Talker contract and conditioning tables."""
    talker = _talker(root)
    config["model"] = ("qwen3_omni_next_talker_text"
                       if cfg.num_experts > 0 else "qwen3_omni_next_talker")
    keys = (
        "tts_pad_token_id",
        "tts_bos_token_id",
        "tts_eos_token_id",
        "codec_nothink_id",
        "codec_think_bos_id",
        "codec_think_eos_id",
        "codec_pad_id",
        "codec_bos_id",
        "codec_eos_token_id",
        "codec_think_id",
        "accept_hidden_layer",
        "num_code_groups",
        "max_thinker_to_talker_mm_tokens",
        "talker_language_id",
        "talker_assistant_prompt_id_mapping",
        "speaker_system_prompt_id",
    )
    for key in keys:
        if key in talker:
            config[key] = talker[key]
        elif key in root:
            config[key] = root[key]

    thinker_text = (_thinker(root).get("text_config") or _thinker(root))
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
    _ = args


def update_code_predictor_config(config: Dict[str, Any], root: Dict[str, Any],
                                 cfg, args) -> None:
    """Write the stacked-head CodePredictor runtime contract."""
    talker = _talker(root)
    config["model"] = "qwen3_omni_next_code_predictor"
    config["use_embeddings_input"] = True
    config["num_code_groups"] = int(talker.get("num_code_groups", 16))
    config["num_deepstack_features"] = 0
    _ = cfg, args


def _vision_config(bundle, args):
    root = bundle.root
    thinker = _thinker(root)
    visual = dict(bundle.component_dict(contracts.Component.VISUAL))
    visual["model_type"] = "qwen3_omni_next_vision_encoder"
    result: Dict[str, Any] = {
        "model_type": "qwen3_omni_next_vision_encoder",
        "vision_config": visual,
    }
    position_rate = thinker.get("position_id_per_seconds",
                                root.get("position_id_per_seconds"))
    if position_rate is not None:
        visual["position_id_per_seconds"] = position_rate
    text = dict(thinker.get("text_config") or root.get("text_config") or {})
    rope = text.get("rope_scaling") or text.get("rope_parameters")
    for key in ("vision_start_token_id", "vision_end_token_id",
                "image_token_id", "video_token_id", "vocab_size",
                "rope_theta"):
        for source in (root, thinker, text):
            if key in source:
                result[key] = source[key]
                break
    if rope:
        normalized = normalize_rope_scaling(rope)
        result["rope_scaling"] = normalized
        text.setdefault("rope_scaling", normalized)
    if text:
        result["text_config"] = text
    result["builder_config"] = {
        "min_image_tokens": args.min_image_tokens,
        "max_image_tokens": args.max_image_tokens,
        "max_image_tokens_per_image": args.max_image_tokens_per_image,
    }
    return result


def _audio_config(bundle, args):
    root = bundle.root
    thinker = _thinker(root)
    audio = dict(bundle.component_dict(contracts.Component.AUDIO))
    result: Dict[str, Any] = {
        "model_type": "qwen3_omni_next_audio_encoder",
        "audio_config": audio,
        "builder_config": {
            "min_time_steps": args.min_time_steps,
            "max_time_steps": args.max_time_steps,
            "min_code_len": args.min_code_len,
            "opt_code_len": args.opt_code_len,
            "max_code_len": args.max_code_len,
        },
    }
    for key in ("audio_token_id", "audio_start_token_id", "audio_end_token_id",
                "user_token_id"):
        if key in thinker:
            result[key] = thinker[key]
        elif key in root:
            result[key] = root[key]
    text = thinker.get("text_config") or root.get("text_config") or {}
    rope_theta = text.get("rope_theta")
    if rope_theta is None:
        rope_theta = (text.get("rope_parameters") or {}).get("rope_theta")
    if rope_theta is not None:
        result["text_config"] = {"rope_theta": rope_theta}
    return result


def _code2wav_config(bundle, args):
    codec = dict(bundle.component_dict(contracts.Component.CODE2WAV))
    return {
        "model_type": "qwen3_omni_next_code2wav",
        "code2wav_config": {
            "num_quantizers": codec["num_quantizers"],
            "codebook_size": codec["codebook_size"],
            "hidden_size": codec["transformer"]["hidden_size"],
            "decoder_dim": codec["decoder_dimension"],
            "upsample_rates": codec["pre_upsample_rates"],
            "upsampling_ratios": codec["decoder_rates"],
            "sample_rate": codec["sample_rate"],
        },
        "builder_config": {
            "min_code_len": args.min_code_len,
            "opt_code_len": args.opt_code_len,
            "max_code_len": args.max_code_len,
        },
    }


def component_runtime_config(bundle, component: contracts.Component, args):
    """Return the exact runtime sidecar for a non-text Next component."""
    if component == contracts.Component.VISUAL:
        return _vision_config(bundle, args)
    if component == contracts.Component.AUDIO:
        return _audio_config(bundle, args)
    if component == contracts.Component.CODE2WAV:
        return _code2wav_config(bundle, args)
    return None
