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
"""Qwen3-Omni checkpoint and component configuration."""

import json
import logging
import os
from dataclasses import replace

from ...core import contracts, quantization, weight_policy

logger = logging.getLogger(__name__)

_TALKER_TYPES = frozenset((
    "qwen3_omni_talker",
    "qwen3_omni_moe_talker",
))
_PREDICTOR_TYPES = frozenset((
    "qwen3_omni_talker_code_predictor",
    "qwen3_omni_moe_talker_code_predictor",
))


def prepare_root(model_dir: str, root: dict) -> dict:
    root = dict(root)
    speech_config = os.path.join(model_dir, "speech_tokenizer", "config.json")
    if os.path.isfile(speech_config):
        with open(speech_config) as config_file:
            root["_speech_tokenizer_config"] = json.load(config_file)
    return root


def component_config(root: dict, component: contracts.Component) -> dict:
    root_type = str(root.get("model_type", ""))
    thinker = root.get("thinker_config") or {}
    if component == contracts.Component.LLM:
        return thinker or root
    if component == contracts.Component.VISUAL:
        return root.get("vision_config") or thinker.get(
            "vision_config") or root
    if component == contracts.Component.AUDIO:
        return root.get("audio_config") or thinker.get("audio_config") or root
    if component == contracts.Component.TALKER:
        talker = root.get("talker_config")
        if isinstance(talker, dict):
            return talker
        if root_type in _TALKER_TYPES:
            return root
        raise ValueError("Qwen3-Omni checkpoint has no talker_config")
    if component == contracts.Component.CODE_PREDICTOR:
        talker = root.get("talker_config") or root
        predictor = talker.get("code_predictor_config")
        if isinstance(predictor, dict):
            return predictor
        if root_type in _PREDICTOR_TYPES:
            return root
        raise ValueError("Qwen3-Omni checkpoint has no code_predictor_config")
    if component == contracts.Component.CODE2WAV:
        talker = root.get("talker_config") or {}
        speech = root.get("_speech_tokenizer_config") or {}
        return (root.get("code2wav_config") or talker.get("code2wav_config")
                or speech.get("decoder_config") or root)
    raise ValueError(f"Qwen3-Omni has no {component.value} configuration")


def component_weight_policy(args, policy):
    """Keep dynamic Code2Wav dense weights out of TensorRT's Myelin path."""
    if args.resolved_component != contracts.Component.CODE2WAV:
        return policy
    return policy.without((weight_policy.EXTERNAL_WEIGHT_FP16, ),
                          strict=bool(args.externalize_weights))


def prepare_text_config(config: dict, root: dict,
                        component: contracts.Component,
                        model_dir: str) -> dict:
    config = dict(config)
    if component == contracts.Component.LLM:
        thinker = root.get("thinker_config") or {}
        visual = thinker.get("vision_config") or root.get(
            "vision_config") or {}
        deepstack = visual.get("deepstack_visual_indexes")
        default_features = (len(deepstack) if isinstance(deepstack, list) else
                            (3 if visual else 0))
        config.setdefault("num_deepstack_features", default_features)
    return config


def configure_base(config, **kwargs) -> None:
    """Enable the native MTP feedback contract on the thinker."""
    config.mtp_base = True


def configure_draft(config, **kwargs) -> None:
    """Select the checkpoint's unquantized full-attention MTP layers."""
    if config.mtp_num_hidden_layers is None:
        raise ValueError("Qwen3-Omni MTP draft requires mtp_num_hidden_layers")
    config.num_hidden_layers = config.mtp_num_hidden_layers
    config.layer_types = ["attention"] * config.num_hidden_layers
    config.attention_layer_types = ["full_attention"
                                    ] * config.num_hidden_layers
    config.gdn_cfg = None
    config.mtp_base = False
    config.tie_word_embeddings = False
    config.quant = replace(config.quant,
                           quant_type=quantization.QUANT_FP16,
                           excluded=(),
                           layer_overrides={},
                           is_mixed_precision=False)


def _accept_hidden_layer(root: dict) -> int:
    talker = root.get("talker_config") or {}
    if talker.get("accept_hidden_layer") is not None:
        return int(talker["accept_hidden_layer"])
    return int(root.get("accept_hidden_layer", -1))


def multistage_topology(bundle, components, args) -> dict | None:
    """Thinker to talker to vocoder, or text-only when the speech half is absent.

    The talker reads the thinker's accept_hidden_layer, so a speech build whose
    checkpoint names no positive layer gets no graph.
    """
    del args
    return omni_topology(bundle,
                         components,
                         prefill_builder="segments",
                         speaks_alone=False)


def _cap_chunked_talker(talker: dict, prefill_builder: str) -> dict:
    """Pin the chunked talker to max_batch 1.

    It feeds text by re-prefilling one member's own KV, which a shared cache
    cannot do for several members at once.
    """
    if prefill_builder == "chunked":
        talker["max_batch"] = 1
    return talker


def omni_topology(bundle, components, prefill_builder: str,
                  speaks_alone: bool) -> dict | None:
    """The Qwen3-Omni graph with the family's talker prefill builder.

    *speaks_alone* says whether the talker runs with no thinker in front. The
    segments talker cannot: it reads the thinker's text embedding from
    ``<engine_dir>/embedding.safetensors``, which only the llm build writes.
    """
    from ...core.artifacts.multistage import (artifact_of, media_inputs,
                                              parsed_request_input)

    present = frozenset(components)
    speech = {
        contracts.Component.TALKER,
        contracts.Component.CODE_PREDICTOR,
        contracts.Component.CODE2WAV,
    }
    if contracts.Component.LLM not in present:
        if not speaks_alone:
            logger.info("No multistage_config.json: the build has no llm "
                        "component, and this talker reads the thinker's "
                        "text embedding")
            return None
        if not speech.issubset(present):
            logger.info("No multistage_config.json: a graph without the "
                        "thinker needs the talker, code predictor and "
                        "code2wav")
            return None
        return _speech_only_topology(prefill_builder)
    thinker_config = {}
    media = media_inputs(present)
    if media:
        # The runtime appends /audio and /visual itself, so this names their parent.
        thinker_config["multimodal_artifact"] = "."
    thinker = {
        "id":
        "thinker",
        "runner":
        "llm.backbone",
        "artifact":
        artifact_of(contracts.Component.LLM),
        "inputs": [{
            "name": "prompt",
            "from": "request:messages",
            "on_missing": "fail"
        }, *media, {
            "name": "sampling",
            "from": "request:sampling",
            "on_missing": "skip"
        },
                   parsed_request_input()],
        "emits": [{
            "id": "text",
            "kind": "text",
            "transport": "chunk_stream",
            "terminal": True
        }],
    }
    if not speech.issubset(present):
        if thinker_config:
            thinker["config"] = thinker_config
        return {"terminals": ["thinker:text"], "stages": [thinker]}

    accept_hidden_layer = _accept_hidden_layer(bundle.root)
    if accept_hidden_layer <= 0:
        # -1 means the checkpoint names no layer; the thinker would then emit
        # its post-norm output, which is not what the talker was trained on.
        # 0 is the input-embedding slot.
        logger.warning(
            "No multistage_config.json: the checkpoint names no positive "
            "accept_hidden_layer for the talker to read")
        return None

    thinker_config["accept_hidden_layer"] = accept_hidden_layer
    thinker["config"] = thinker_config
    thinker["emits"].append({
        "id": "accept_hidden",
        "kind": "hidden",
        "transport": "chunk_stream"
    })
    # The prompt half of the token stream, after the chat template. The talker's prefill
    # builders locate the assistant segment in it; the generated tail alone has no segment.
    thinker["emits"].append({
        "id": "prompt_ids",
        "kind": "text",
        "transport": "chunk_stream"
    })
    talker = {
        "id":
        "talker",
        "runner":
        "qwen3_omni.talker",
        "artifact":
        artifact_of(contracts.Component.TALKER),
        "inputs": [
            {
                "name": "hidden",
                "from": "thinker:accept_hidden",
                "on_missing": "fail"
            },
            # The talker speaks what the thinker wrote, so it needs the tokens
            # as well as the hidden states behind them.
            {
                "name": "prompt_ids",
                "from": "thinker:prompt_ids",
                "on_missing": "fail"
            },
            {
                "name": "text_ids",
                "from": "thinker:text",
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
            "prefill_builder":
            prefill_builder,
            # Resolved against the talker's own directory; the builder makes the
            # code predictor its sibling.
            "code_predictor_artifact":
            "../" + artifact_of(contracts.Component.CODE_PREDICTOR),
            # Verify window for code predictor speculation; 0 leaves the
            # autoregressive RVQ loop in place.
            "cp_spec_verify_size":
            0,
        },
    }
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
    return {
        "terminals": ["thinker:text", "code2wav:audio"],
        "stages":
        [thinker,
         _cap_chunked_talker(talker, prefill_builder), code2wav],
    }


def _speech_only_topology(prefill_builder: str) -> dict:
    """Speak a written turn with no thinker in front.

    Without a backbone there are no hidden states to condition on, so the talker renders the
    conversation itself.
    """
    from ...core.artifacts.multistage import artifact_of

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
            "prefill_builder":
            prefill_builder,
            "code_predictor_artifact":
            "../" + artifact_of(contracts.Component.CODE_PREDICTOR),
            "cp_spec_verify_size":
            0,
        },
    }
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
    return {
        "terminals": ["code2wav:audio"],
        "stages": [_cap_chunked_talker(talker, prefill_builder), code2wav]
    }
