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
"""
CLI: export ALL components of a multimodal checkpoint to ONNX in one command.

Detects model type from ``config.json`` and exports:
    - LLM backbone        → ``<output_dir>/llm/model.onnx``
    - Visual encoder      → ``<output_dir>/visual/model.onnx``    (VLMs)
    - Audio encoder       → ``<output_dir>/audio/model.onnx``     (speech models)
    - Code2Wav vocoder    → ``<output_dir>/code2wav/model.onnx``  (Qwen3-Omni / Qwen3-TTS)

Usage::

    # From a source checkout or installed package:
    tensorrt-edgellm-export /path/to/checkpoint /tmp/onnx_out

    # With explicit dtype
    tensorrt-edgellm-export /path/to/Qwen3-VL-7B /tmp/out --dtype float16

    # With reduced vocabulary
    tensorrt-edgellm-export /path/to/model /tmp/out --reduced-vocab-dir /path/to/reduced_vocab

Supported model types
----------------------
VLMs (LLM + visual encoder):
    qwen3_vl, qwen3_omni          (Qwen3-VL / Qwen3-Omni)
    qwen3_5, qwen3_5_moe          (Qwen3.5)
    qwen2_5_vl                    (Qwen2.5-VL)
    internvl_chat                 (InternVL3)
    internvl                      (InternVL3.5)
    phi4mm, phi4_multimodal       (Phi-4 Multimodal)
    gemma4, gemma4_unified        (Gemma4 multimodal checkpoints)
    NemotronH_Nano_VL_V2, NemotronH_Nano_Omni_Reasoning_V3
                                 (Nemotron-Omni)

Audio models (LLM + audio encoder):
    qwen3_asr, qwen3_omni, qwen3_omni_thinker, gemma4_unified
    NemotronH_Nano_VL_V2, NemotronH_Nano_Omni_Reasoning_V3
                                 (Nemotron-Omni)

LLM + Talker decoder + Code2Wav (no audio encoder):
    qwen3_tts    (Talker/CodePredictor are LLM decoders; Code2Wav uses speech_tokenizer/)

LLM-only:
    All other model types supported by :mod:`tensorrt_edgellm.model.AutoModel`.
"""

import argparse
import json
import logging
import os
import sys
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from ..config import ModelConfig

# Importing the CLI module no longer eagerly loads the package export API.
# Register model-family implementations before AutoModel dispatch below.
from .. import _export_api as _registered_export_api  # noqa: F401
from ..checkpoint.checkpoint_utils import normalize_rope_scaling_for_runtime
from ..config import _is_diffusion_gemma_model_type
from ..external_weights import (EXTERNAL_WEIGHT_CHOICES,
                                EXTERNAL_WEIGHT_NVFP4_MOE,
                                resolve_externalize_weights)
from ..models.ops import set_int4_gemm_plugin_version

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tensorrt_edgellm.scripts.export")

# ---------------------------------------------------------------------------
# Model type classification
# ---------------------------------------------------------------------------

_NEMOTRON_OMNI_MODEL_TYPES = frozenset([
    "NemotronH_Nano_VL_V2",
    "NemotronH_Nano_Omni_Reasoning_V3",
])

_GEMMA4_MODEL_TYPES = frozenset([
    "gemma4",
    "gemma4_text",
    "gemma4_unified",
    "gemma4_unified_text",
])

_GEMMA4_ASSISTANT_MODEL_TYPES = frozenset([
    "gemma4_assistant",
    "gemma4_unified_assistant",
])

_VLM_MODEL_TYPES = frozenset([
    "qwen3_vl",
    "qwen3_omni",
    "qwen3_omni_moe",
    "qwen3_omni_next",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen2_5_vl",
    "internvl",
    "internvl_chat",
    "phi4mm",
    "phi4_multimodal",
    "gemma4",
    "gemma4_unified",
    "alpamayo_r1",
    *_NEMOTRON_OMNI_MODEL_TYPES,
])

_AUDIO_MODEL_TYPES = frozenset([
    "nemotron3_5_asr",
    "gemma4",
    "qwen3_asr",
    "qwen3_omni",
    "qwen3_omni_thinker",
    "qwen3_omni_moe",
    "qwen3_omni_moe_thinker",
    "qwen3_omni_next",
    "qwen3_omni_next_thinker",
    "gemma4_unified",
    *_NEMOTRON_OMNI_MODEL_TYPES,
    # qwen3_tts intentionally excluded: Qwen3-TTS has NO audio encoder.
    # Its Talker and CodePredictor are LLM decoders exported via the LLM pipeline.
])

# Subset of audio models whose LLM config follows the Qwen-style
# ``thinker_config`` layout for modality-token IDs and chat-template tokens.
# Excludes Nemotron-Omni, which has its own field names
# (``img_context_token_id`` / ``sound_context_token_id``) at the source-config
# root and is handled by ``_collect_tokens_from_nemotron_root``.
# ``nemotron3_5_asr`` is subtracted too: it is a pure RNN-T ASR model with no
# LLM decoder at all (no ``thinker_config``, no ``user_token_id``), so the
# Qwen-style multimodal-token patching does not apply.
_ASR_LLM_MODEL_TYPES = (_AUDIO_MODEL_TYPES - _NEMOTRON_OMNI_MODEL_TYPES -
                        {"gemma4_unified", "nemotron3_5_asr"})

_CODE2WAV_MODEL_TYPES = frozenset([
    "qwen3_omni",
    "qwen3_omni_moe",
    "qwen3_omni_next",
    "qwen3_tts",
])

_ACTION_MODEL_TYPES = frozenset([
    "alpamayo_r1",
])
# Which LLM-family components each model ships.  Default (unlisted model types)
# is ``{"thinker"}``.  Add a new Talker/CP-bearing model by listing it here; no
# other bookkeeping in this file is needed for component dispatch.
_LLM_COMPONENTS: dict[str, frozenset[str]] = {
    "qwen3_tts": frozenset(["talker", "code_predictor"]),  # no thinker
    "qwen3_omni": frozenset(["thinker", "talker", "code_predictor"]),
    "qwen3_omni_moe": frozenset(["thinker", "talker", "code_predictor"]),
    "qwen3_omni_next": frozenset(["thinker", "talker", "code_predictor"]),
    # Talker-root checkpoint from the CP-FP8 quantization pass
    # (``--cp_quantization fp8``); only the CodePredictor is re-exported
    # from it — thinker/talker/encoders come from the original HF root.
    "qwen3_omni_next_talker": frozenset(["code_predictor"]),
    "nemotron3_5_asr": frozenset(),  # pure RNN-T transducer
}
_DEFAULT_LLM_COMPONENTS = frozenset(["thinker"])


def _has_visual(model_type: str) -> bool:
    return model_type in _VLM_MODEL_TYPES


def _has_audio(model_type: str) -> bool:
    return model_type in _AUDIO_MODEL_TYPES


def _has_rnnt_decoder(model_type: str) -> bool:
    """Whether ``model_type`` has an RNN-T (transducer) decoder-step engine
    exported alongside its encoder — currently only Nemotron-3.5-ASR."""
    return model_type == "nemotron3_5_asr"


def _checkpoint_audio_config(config: dict) -> "dict | None":
    """Locate the audio-encoder config wherever the checkpoint stores it.

    Gemma4 / Gemma4-Unified keep ``audio_config`` at the root; Qwen3-ASR /
    Qwen3-Omni nest it under ``thinker_config``; Nemotron-Omni names it
    ``sound_config``; Nemotron-3.5-ASR names it ``encoder_config``. Returns
    ``None`` when the checkpoint genuinely has no audio encoder (e.g. Gemma4
    dense with ``"audio_config": null``).
    """
    return (config.get("audio_config")
            or (config.get("thinker_config") or {}).get("audio_config")
            or config.get("sound_config") or config.get("encoder_config"))


def _has_action(model_type: str) -> bool:
    return model_type in _ACTION_MODEL_TYPES


def _has_code2wav(model_type: str) -> bool:
    return model_type in _CODE2WAV_MODEL_TYPES


def _has_llm_component(model_type: str, component: str) -> bool:
    """Whether ``model_type`` exports ``component`` as part of its LLM family.

    ``component`` is one of ``"thinker"``, ``"talker"``, ``"code_predictor"``.
    """
    return component in _LLM_COMPONENTS.get(model_type,
                                            _DEFAULT_LLM_COMPONENTS)


def _is_alpamayo(model_type: str) -> bool:
    return model_type == "alpamayo_r1"


def _is_diffusion_gemma(model_type: str, config: dict) -> bool:
    architectures = [str(a).lower() for a in config.get("architectures", [])]
    return (_is_diffusion_gemma_model_type(model_type)
            or any("diffusiongemma" in arch for arch in architectures))


# Default output sub-path for every component.  Model types that need a
# non-default path only list the components that differ in ``_LAYOUT_OVERRIDES``.
_DEFAULT_LAYOUT: dict[str, str] = {
    "thinker": "llm",
    "talker": "talker",
    "code_predictor": "code_predictor",
    "audio": "audio",
    "rnnt_decoder": "rnnt_decoder",
    "code2wav": "code2wav",
    "visual": "visual",
    "action": "action",
    "mtp_draft": "mtp_draft",
    "dflash_draft": "dflash_draft",
    "jetspec_draft": "jetspec_draft",
    "dspark_draft": "dspark_draft",
}

# Per-model overrides on top of ``_DEFAULT_LAYOUT``.
_LAYOUT_OVERRIDES: dict[str, dict[str, str]] = {
    "qwen3_omni": {
        "thinker": "llm/thinker",
        "talker": "llm/talker",
        "code_predictor": "llm/code_predictor",
        "audio": "audio/audio_encoder",
        "code2wav": "audio/code2wav",
        "visual": "vision",
    },
    # Qwen3-Next Omni reuses the same on-disk layout as Qwen3-Omni so existing
    # engine-build / runtime scripts that expect ``llm/thinker``,
    # ``llm/talker``, ``audio/audio_encoder``, etc. keep working unchanged.
    "qwen3_omni_next": {
        "thinker": "llm/thinker",
        "talker": "llm/talker",
        "code_predictor": "llm/code_predictor",
        "audio": "audio/audio_encoder",
        "code2wav": "audio/code2wav",
        "visual": "vision",
    },
    # ``qwen3_omni_moe`` shares the Omni ONNX layout.
    "qwen3_omni_moe": {
        "thinker": "llm/thinker",
        "talker": "llm/talker",
        "code_predictor": "llm/code_predictor",
        "audio": "audio/audio_encoder",
        "code2wav": "audio/code2wav",
        "visual": "vision",
    },
    "qwen3_omni_next_talker": {
        "code_predictor": "llm/code_predictor",
    },
    "qwen3_tts": {
        # Qwen3-TTS has no thinker; the Talker is written under ``llm/`` so
        # existing engine-build scripts that expect a single ``llm/`` dir
        # keep working.
        "talker": "llm",
    },
}


def _layout_for(model_type: str, component: str) -> str:
    """Return the output sub-path for ``component`` under the export root."""
    return _LAYOUT_OVERRIDES.get(model_type,
                                 {}).get(component, _DEFAULT_LAYOUT[component])


# Helpers
# ---------------------------------------------------------------------------


def _resolve_model_dir(model: str) -> str:
    if os.path.isdir(model):
        return model
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub is not installed. "
                     "Install it or provide a local path.")
        sys.exit(1)
    logger.info("Downloading %s from Hugging Face Hub ...", model)
    return snapshot_download(model)


def _resolve_mtp_draft_dir(mtp_draft_dir: str) -> str:
    """Resolve a user-provided paired MTP draft checkpoint path or HF repo ID."""
    if os.path.isdir(mtp_draft_dir):
        return os.path.abspath(mtp_draft_dir)
    if os.path.isabs(mtp_draft_dir) or mtp_draft_dir.startswith("."):
        return os.path.abspath(mtp_draft_dir)
    return _resolve_model_dir(mtp_draft_dir)


def _load_config(model_dir: str) -> dict:
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        logger.error("config.json not found in %s", model_dir)
        sys.exit(1)
    with open(cfg_path) as f:
        return json.load(f)


def _is_cosmos3_checkpoint(model_dir: str) -> bool:
    """Detect a Cosmos3 diffusers checkpoint via ``model_index.json``."""
    index_path = os.path.join(model_dir, "model_index.json")
    if not os.path.exists(index_path):
        return False
    try:
        with open(index_path) as f:
            class_name = json.load(f).get("_class_name", "")
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(class_name, str) and class_name.startswith("Cosmos3")


def _get_llm_text_config(config: dict) -> dict:
    """Return the promoted text/LLM config dict when present."""
    for key in ("text_config", "llm_config", "language_config"):
        sub = config.get(key)
        if isinstance(sub, dict) and sub.get("hidden_size") is not None:
            return sub
    return config


def _has_mtp(config: dict) -> bool:
    """Return True when the checkpoint exposes the MTP branch."""
    # Omni checkpoints nest the LLM config one level deeper
    # (thinker_config.text_config).
    thinker = config.get("thinker_config")
    if isinstance(thinker, dict):
        sub = thinker.get("text_config")
        if isinstance(sub,
                      dict) and sub.get("mtp_num_hidden_layers") is not None:
            return True
    text_cfg = _get_llm_text_config(config)
    if text_cfg.get("mtp_num_hidden_layers") is not None:
        return True
    # Nemotron-H (DeepSeek-V3 naming): one or more MTP prediction modules whose
    # layer stack is given by ``mtp_hybrid_override_pattern`` or, on newer
    # checkpoints, the ``mtp_layers_block_type`` list.
    return bool(
        int(text_cfg.get("num_nextn_predict_layers", 0) or 0) > 0
        and (text_cfg.get("mtp_hybrid_override_pattern")
             or text_cfg.get("mtp_layers_block_type")))


def _normalize_gemma4_layer_type(layer_type: str) -> str:
    if layer_type in ("sliding_attention", "full_attention"):
        return layer_type
    raise ValueError(
        f"Unsupported Gemma4 assistant layer type {layer_type!r}; "
        "expected sliding_attention or full_attention.")


def _assert_tokenizers_match(target_dir: str, assistant_dir: str) -> None:

    def _token_id_map(tokenizer_path: str) -> dict:
        with open(tokenizer_path) as f:
            tokenizer = json.load(f)
        token_ids = {}
        vocab = tokenizer.get("model", {}).get("vocab", {})
        if isinstance(vocab, dict):
            token_ids.update({
                str(token): int(idx)
                for token, idx in vocab.items()
            })
        for entry in tokenizer.get("added_tokens", []):
            content = entry.get("content")
            idx = entry.get("id")
            if content is not None and idx is not None:
                token_ids[str(content)] = int(idx)
        return token_ids

    target_tok = os.path.join(target_dir, "tokenizer.json")
    assistant_tok = os.path.join(assistant_dir, "tokenizer.json")
    if not os.path.exists(target_tok) or not os.path.exists(assistant_tok):
        raise ValueError(
            "Gemma4 MTP pairing requires tokenizer.json in both target and assistant checkpoints."
        )
    if _token_id_map(target_tok) != _token_id_map(assistant_tok):
        raise ValueError(
            "Gemma4 MTP target and assistant tokenizer token-id maps differ; "
            "paired export requires identical token IDs.")


def _build_gemma4_kv_sharing_map(target_config: dict,
                                 assistant_config: dict) -> list[dict]:
    target_text = _get_llm_text_config(target_config)
    assistant_text = _get_llm_text_config(assistant_config)
    target_types = [
        _normalize_gemma4_layer_type(str(layer_type))
        for layer_type in target_text.get("layer_types", [])
    ]
    assistant_types = [
        _normalize_gemma4_layer_type(str(layer_type))
        for layer_type in assistant_text.get("layer_types", [])
    ]
    num_shared_target_layers_value = target_text.get("num_kv_shared_layers")
    num_shared_target_layers = (len(target_types)
                                if num_shared_target_layers_value is None else
                                int(num_shared_target_layers_value))
    num_shared_target_layers = min(num_shared_target_layers, len(target_types))
    target_share_start = len(target_types) - num_shared_target_layers
    target_donor_types = target_types[:target_share_start]

    kv_sharing_map = []
    # Gemma4 assistant layers read target KV from the nearest compatible donor
    # before the shared-KV suffix. Layers with the same attention type may
    # intentionally share the same target donor.
    for assistant_layer, assistant_type in enumerate(assistant_types):
        matched_target = None
        for target_layer in range(len(target_donor_types) - 1, -1, -1):
            if target_types[target_layer] == assistant_type:
                matched_target = target_layer
                break
        if matched_target is None:
            raise ValueError(
                "Gemma4 MTP assistant layer %d (%s) cannot be mapped to a "
                "compatible donor layer before the target shared-KV suffix of %d layers."
                % (assistant_layer, assistant_type, num_shared_target_layers))
        kv_sharing_map.append({
            "assistant_layer": assistant_layer,
            "target_attention_layer": matched_target,
            "target_layer": matched_target,
            "target_layer_type": assistant_type,
        })
    return kv_sharing_map


def _gemma4_head_dim_for_layer(text_config: dict, layer_type: str) -> int:
    if layer_type == "full_attention" and text_config.get("global_head_dim"):
        return int(text_config["global_head_dim"])
    return int(text_config.get("head_dim", 0))


def _gemma4_num_kv_heads_for_layer(text_config: dict, layer_type: str) -> int:
    if (layer_type == "full_attention" and text_config.get("attention_k_eq_v")
            and text_config.get("num_global_key_value_heads")):
        return int(text_config["num_global_key_value_heads"])
    return int(text_config.get("num_key_value_heads", 0))


def _validate_gemma4_kv_sharing_contract(target_config: dict,
                                         assistant_config: dict,
                                         kv_sharing_map: list[dict]) -> None:
    target_text = _get_llm_text_config(target_config)
    assistant_text = _get_llm_text_config(assistant_config)
    target_types = [
        _normalize_gemma4_layer_type(str(layer_type))
        for layer_type in target_text.get("layer_types", [])
    ]
    assistant_types = [
        _normalize_gemma4_layer_type(str(layer_type))
        for layer_type in assistant_text.get("layer_types", [])
    ]
    if len(assistant_types) != int(assistant_text.get("num_hidden_layers", 0)):
        raise ValueError(
            "Gemma4 MTP assistant layer_types length must match num_hidden_layers."
        )
    if len(target_types) != int(target_text.get("num_hidden_layers", 0)):
        raise ValueError(
            "Gemma4 MTP target layer_types length must match num_hidden_layers."
        )
    if len(kv_sharing_map) != len(assistant_types):
        raise ValueError(
            "Gemma4 MTP kv_sharing_map length must match assistant layer count."
        )

    seen_layers = set()
    for entry in kv_sharing_map:
        assistant_layer = int(entry["assistant_layer"])
        target_layer = int(entry["target_attention_layer"])
        if assistant_layer in seen_layers:
            raise ValueError(
                "Gemma4 MTP kv_sharing_map has duplicate assistant layer %d." %
                assistant_layer)
        seen_layers.add(assistant_layer)
        if assistant_layer < 0 or assistant_layer >= len(assistant_types):
            raise ValueError(
                "Gemma4 MTP assistant layer %d is outside the assistant layer range."
                % assistant_layer)
        if target_layer < 0 or target_layer >= len(target_types):
            raise ValueError(
                "Gemma4 MTP target donor layer %d is outside the target layer range."
                % target_layer)

        assistant_type = assistant_types[assistant_layer]
        target_type = target_types[target_layer]
        if assistant_type != target_type:
            raise ValueError(
                "Gemma4 MTP layer type mismatch: assistant layer %d is %s, target donor layer %d is %s."
                % (assistant_layer, assistant_type, target_layer, target_type))

        assistant_head_dim = _gemma4_head_dim_for_layer(
            assistant_text, assistant_type)
        target_head_dim = _gemma4_head_dim_for_layer(target_text, target_type)
        if assistant_head_dim != target_head_dim:
            raise ValueError(
                "Gemma4 MTP KV head_dim mismatch: assistant layer %d head_dim=%d, target donor layer %d head_dim=%d."
                % (assistant_layer, assistant_head_dim, target_layer,
                   target_head_dim))

        assistant_kv_heads = _gemma4_num_kv_heads_for_layer(
            assistant_text, assistant_type)
        target_kv_heads = _gemma4_num_kv_heads_for_layer(
            target_text, target_type)
        if assistant_kv_heads != target_kv_heads:
            raise ValueError(
                "Gemma4 MTP KV head mismatch: assistant layer %d num_kv_heads=%d, target donor layer %d num_kv_heads=%d."
                % (assistant_layer, assistant_kv_heads, target_layer,
                   target_kv_heads))

    expected_layers = set(range(len(assistant_types)))
    if seen_layers != expected_layers:
        missing = sorted(expected_layers - seen_layers)
        raise ValueError(
            "Gemma4 MTP kv_sharing_map is missing assistant layers: %s." %
            missing)


def _validate_gemma4_mtp_pair(target_dir: str,
                              assistant_dir: str) -> list[dict]:
    target_config = _load_config(target_dir)
    assistant_config = _load_config(assistant_dir)
    target_text = _get_llm_text_config(target_config)
    assistant_text = _get_llm_text_config(assistant_config)

    if target_text.get("model_type") not in _GEMMA4_MODEL_TYPES:
        raise ValueError(
            "Gemma4 MTP target must have a Gemma4 model_type in text_config.")
    if assistant_config.get("model_type") not in _GEMMA4_ASSISTANT_MODEL_TYPES:
        raise ValueError(
            "Gemma4 MTP assistant must have a Gemma4 assistant root model_type."
        )
    if int(target_text.get("hidden_size", 0)) != int(
            assistant_config.get("backbone_hidden_size", 0)):
        raise ValueError(
            "Gemma4 MTP hidden mismatch: target hidden_size=%s, assistant backbone_hidden_size=%s"
            % (target_text.get("hidden_size"),
               assistant_config.get("backbone_hidden_size")))
    if int(target_text.get("vocab_size",
                           0)) != int(assistant_text.get("vocab_size", 0)):
        raise ValueError(
            "Gemma4 MTP vocab mismatch: target vocab_size=%s, assistant vocab_size=%s"
            %
            (target_text.get("vocab_size"), assistant_text.get("vocab_size")))
    if int(assistant_text.get("num_kv_shared_layers", 0)) != int(
            assistant_text.get("num_hidden_layers", 0)):
        raise ValueError(
            "Gemma4 MTP assistant requires num_kv_shared_layers == num_hidden_layers."
        )
    _assert_tokenizers_match(target_dir, assistant_dir)
    kv_sharing_map = _build_gemma4_kv_sharing_map(target_config,
                                                  assistant_config)
    _validate_gemma4_kv_sharing_contract(target_config, assistant_config,
                                         kv_sharing_map)
    return kv_sharing_map


def _find_token_id(model_dir: str, token_str: str) -> "Optional[int]":
    """Return the token ID for *token_str* by scanning tokenizer files."""
    # tokenizer.json: scan ``added_tokens`` and ``added_tokens_decoder`` in a
    # single open. Qwen3-Omni / VL-style ckpts ship the special-token IDs here.
    tok_path = os.path.join(model_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        with open(tok_path) as f:
            tok = json.load(f)
        for entry in tok.get("added_tokens", []):
            if entry.get("content") == token_str:
                return int(entry["id"])
        for id_str, entry in tok.get("added_tokens_decoder", {}).items():
            if entry.get("content") == token_str:
                return int(id_str)
    # added_tokens.json: older HF tokenizer split-file form.
    added_path = os.path.join(model_dir, "added_tokens.json")
    if os.path.exists(added_path):
        with open(added_path) as f:
            added = json.load(f)
        if token_str in added:
            return int(added[token_str])
    # vocab.json: BPE base vocabulary (regular words, not special tokens).
    # Required for Qwen3-ASR ckpts which ship only ``vocab.json`` +
    # ``merges.txt`` (no ``tokenizer.json``), so the literal word ``"user"``
    # has to be resolved here.
    vocab_path = os.path.join(model_dir, "vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            vocab = json.load(f)
        if token_str in vocab:
            return int(vocab[token_str])
    logger.warning("Could not find token ID for %r in %s", token_str,
                   model_dir)
    return None


def _load_all_weights(model_dir: str) -> dict:
    """Load all safetensors (or PyTorch ``.bin``) shards in *model_dir* into a
    flat dict."""
    import glob

    from safetensors.torch import load_file

    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if shards:
        weights: dict = {}
        for shard in shards:
            logger.info("  Loading shard: %s", os.path.basename(shard))
            weights.update(load_file(shard, device="cpu"))
        return weights

    # Fallback: PyTorch pickle checkpoints. Some officially supported AWQ VLMs
    # (e.g. InternVL3-*-AWQ) ship only ``pytorch_model*.bin`` and no safetensors.
    # Mirror the ``.bin`` support the core weight loader already
    # has (checkpoint/loader.py _build_shard_map).
    import torch

    bin_shards = sorted(glob.glob(os.path.join(model_dir, "pytorch_model*.bin"))) \
        or sorted(glob.glob(os.path.join(model_dir, "*.bin")))
    if not bin_shards:
        logger.error("No safetensors or .bin weight files found in %s",
                     model_dir)
        sys.exit(1)
    weights = {}
    for shard in bin_shards:
        logger.info("  Loading .bin shard: %s", os.path.basename(shard))
        weights.update(torch.load(shard, map_location="cpu",
                                  weights_only=True))
    return weights


def _to_fp16(tensors: dict) -> dict:
    """Cast bfloat16 tensors to float16 for C++ runtime compatibility.

    The C++ runtime requires FP16 (or FP8) for sidecar weight files.
    Checkpoints often store weights in bfloat16.
    """
    import torch
    return {
        k: v.to(torch.float16) if v.dtype == torch.bfloat16 else v
        for k, v in tensors.items()
    }


def _dtype_from_str(s: str) -> "torch.dtype":
    import torch
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if s not in mapping:
        logger.error("Unknown dtype %r. Choose from: %s", s,
                     ", ".join(mapping))
        sys.exit(1)
    return mapping[s]


# ---------------------------------------------------------------------------
# Multimodal special-token ID resolution
#
# The C++ runtime (``llmEngineRunner.cpp``) reads ``audio_token_id`` and
# ``image_token_id`` from the LLM's ``config.json`` to find placeholder
# positions that must be replaced with encoder embeddings.  When those fields
# are 0 (default), the runtime skips substitution and the model generates text
# like "I can't hear audio".
#
# Different checkpoint families keep the IDs in different places.  Each
# family has its own collector; ``_patch_multimodal_token_ids`` picks the
# right one based on ``model_type``.
# ---------------------------------------------------------------------------

_TOKEN_KEYS = (
    "audio_token_id",
    "audio_start_token_id",
    "audio_end_token_id",
    "image_token_id",
    "video_token_id",
    "vision_start_token_id",
    "vision_end_token_id",
)


def _collect_tokens_from_thinker_config(root: dict) -> dict:
    """Qwen3-Omni / Qwen3-ASR: IDs live under ``thinker_config`` (with root
    as fallback for flatter variants)."""
    thinker_cfg = root.get("thinker_config") or {}
    out: dict = {}
    for key in _TOKEN_KEYS:
        value = thinker_cfg.get(key, root.get(key))
        if isinstance(value, int):
            out[key] = value
    return out


def _collect_tokens_from_nemotron_root(root: dict) -> dict:
    """Nemotron-Omni: checkpoint-local names at the root level, renamed to
    the runtime's canonical names."""
    rename = {
        "img_context_token_id": "image_token_id",
        "sound_context_token_id": "audio_token_id",
    }
    out: dict = {}
    for src_key, dst_key in rename.items():
        v = root.get(src_key)
        if isinstance(v, int):
            out[dst_key] = v
    return out


def _collect_tokens_from_tokenizer_fallback(model_dir: str) -> dict:
    """Generic VLM fallback: resolve the image placeholder by tokenizing its
    special-token string.  Used when no structured config exposes the ID."""
    out: dict = {}
    image_id = _find_token_id(model_dir, "<|image_pad|>")
    if image_id is not None:
        out["image_token_id"] = image_id
    return out


def _collect_gemma4_tokenizer_fallback(model_dir: str,
                                       model_type: str = "") -> dict:
    """Gemma4 fallback for multimodal placeholders.

    Gemma4's PLE preprocessor uses image/audio token IDs to zero-fill the token
    identity component at multimodal positions. Prefer structured config fields
    when present, but resolve the standard placeholder tokens from tokenizer
    assets when the source config is flat or incomplete.  Gemma4 Unified
    checkpoints name the placeholders ``<|image|>`` / ``<|audio|>``.
    """
    unified = str(model_type).startswith("gemma4_unified")
    image_tokens = ("<|image|>",
                    "<|image_pad|>") if unified else ("<|image_pad|>", )
    audio_tokens = ("<|audio|>",
                    "<|audio_pad|>") if unified else ("<|audio_pad|>", )
    out: dict = {}
    for token in image_tokens:
        image_id = _find_token_id(model_dir, token)
        if image_id is not None:
            out["image_token_id"] = image_id
            break
    for token in audio_tokens:
        audio_id = _find_token_id(model_dir, token)
        if audio_id is not None:
            out["audio_token_id"] = audio_id
            break
    return out


def _collect_user_token_id(model_dir: str, root: dict) -> dict:
    """Resolve ``user_token_id`` for Qwen3-ASR / Qwen3-Omni.

    Qwen3-ASR's source HF config does not expose this field at the
    ``thinker_config`` or root level. When the config does not carry it, fall
    back to a BPE ``vocab.json`` lookup of the literal ``"user"`` word.
    """
    thinker_cfg = root.get("thinker_config") or {}
    v = thinker_cfg.get("user_token_id", root.get("user_token_id"))
    if isinstance(v, int):
        return {"user_token_id": v}
    v = _find_token_id(model_dir, "user")
    if v is not None:
        return {"user_token_id": v}
    return {}


def _patch_multimodal_token_ids(model_dir: str,
                                llm_out_dir: str,
                                model_type: str,
                                config_filename: str = "config.json") -> None:
    """Inject multimodal special-token IDs into the exported LLM runtime config.

    Dispatches to the family-specific collector in priority order:
    thinker_config → Nemotron rename → tokenizer fallback.  Any IDs a
    later collector returns that aren't already set by an earlier one are
    merged in (handles hybrid layouts).
    """
    cfg_path = os.path.join(llm_out_dir, config_filename)
    if not os.path.exists(cfg_path):
        return

    root = _load_config(model_dir)
    collected: dict = {}

    # Primary collectors keyed by model family.
    if model_type in _NEMOTRON_OMNI_MODEL_TYPES:
        collected.update(_collect_tokens_from_nemotron_root(root))
    else:
        collected.update(_collect_tokens_from_thinker_config(root))

    # Historical fallback: any VLM still missing image_token_id — resolve
    # from the tokenizer's ``<|image_pad|>`` special token.
    if "image_token_id" not in collected and model_type in _VLM_MODEL_TYPES:
        collected.update(_collect_tokens_from_tokenizer_fallback(model_dir))
        if "image_token_id" not in collected:
            collected.update(
                _collect_tokens_from_tokenizer_fallback(llm_out_dir))

    # Phi-4 Multimodal names its image placeholder ``<|endoftext10|>`` (id
    # 200010) rather than ``<|image_pad|>``, so the generic fallback above does
    # not find it. The runtime needs image_token_id in the LLM config to insert
    # visual embeddings at those positions (there is no image_token_id field in
    # the source config), so resolve it from the tokenizer here.
    if "image_token_id" not in collected and model_type in ("phi4mm",
                                                            "phi4_multimodal"):
        image_id = _find_token_id(model_dir, "<|endoftext10|>")
        if image_id is None:
            image_id = _find_token_id(llm_out_dir, "<|endoftext10|>")
        if image_id is not None:
            collected["image_token_id"] = image_id

    # Gemma4 PLE needs multimodal placeholder IDs for zero-filling PLE token
    # identity at image/audio positions.
    if model_type in _GEMMA4_MODEL_TYPES:
        fallback = _collect_gemma4_tokenizer_fallback(llm_out_dir, model_type)
        fallback.update(
            _collect_gemma4_tokenizer_fallback(model_dir, model_type))
        for key in ("image_token_id", "audio_token_id"):
            if key not in collected and key in fallback:
                collected[key] = fallback[key]

    # Qwen3-ASR / Qwen3-Omni metadata: ``user_token_id`` (with a BPE vocab
    # fallback) plus the ``model: "qwen3asrthinker"`` tag.
    if model_type in _ASR_LLM_MODEL_TYPES:
        collected.update(_collect_user_token_id(model_dir, root))
        if model_type == "qwen3_asr":
            collected["model"] = "qwen3asrthinker"

    if not collected:
        return

    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg.update(collected)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    pretty = ", ".join(f"{k}={v}" for k, v in collected.items())
    logger.info("[LLM] Patched multimodal token IDs: %s", pretty)


# ---------------------------------------------------------------------------
# Export stages
# ---------------------------------------------------------------------------


def _is_nvfp4_checkpoint(model_dir: str) -> bool:
    """Return True if *model_dir* contains an NVFP4-quantized checkpoint.

    Checks ``hf_quant_config.json`` and ``config.json`` for FP4/NVFP4
    quantization indicators.
    """
    for fname in ("hf_quant_config.json", "config.json"):
        path = os.path.join(model_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        # hf_quant_config.json nests under "quantization"
        quant_section = cfg.get("quantization", {})
        # config.json embeds under "quantization_config"
        qcfg = cfg.get("quantization_config", {})
        # Check all known algo key names across both sections
        for section in (cfg, quant_section, qcfg):
            for key in ("algorithm", "quant_algo", "quant_method"):
                algo = section.get(key, "")
                if isinstance(algo, str) and "FP4" in algo.upper():
                    return True
    return False


def _diffusion_gemma_backbone_externalize_weights(
        model_dir: str,
        externalize_weights: "list[str] | None") -> "list[str]":
    """Return externalized weight kinds required by DiffusionGemma backbone."""
    backbone_externalize_weights = list(externalize_weights or [])
    if (_is_nvfp4_checkpoint(model_dir)
            and EXTERNAL_WEIGHT_NVFP4_MOE not in backbone_externalize_weights):
        backbone_externalize_weights.append(EXTERNAL_WEIGHT_NVFP4_MOE)
        logger.info("[DiffusionGemma] Auto-enabling NVFP4 MoE external "
                    "weights for the backbone export")
    return backbone_externalize_weights


def _alpamayo_llm_key_remap(key: str) -> "Optional[str]":
    """Remap ``vlm.lm_head.*`` → ``lm_head.*`` (not covered by prefix detection)."""
    if key.startswith("vlm.lm_head."):
        return key[len("vlm."):]
    return key


def _cosmos3_edge_llm_key_remap(key: str) -> "Optional[str]":
    """Map the Cosmos3-Edge native reasoner schema onto the default ``CausalLM``.

    The root ``model.safetensors`` stores the text tower flat (``layers.N.*``,
    ``embed_tokens.weight``, ``norm.weight``, ``lm_head.weight``) with Qwen-VL
    style attention names (``to_q/to_k/to_v/to_out``); the vision tower
    (``model.visual.*`` / ``model.projector.*``) is exported separately, and the
    per-layer ``k_norm_und_for_gen`` belongs only to the GEN diffusion tower.
    """
    if "visual." in key or "projector." in key or "k_norm_und_for_gen" in key:
        return None
    for src, dst in ((".self_attn.to_q.", ".self_attn.q_proj."),
                     (".self_attn.to_k.", ".self_attn.k_proj."),
                     (".self_attn.to_v.", ".self_attn.v_proj."),
                     (".self_attn.to_out.", ".self_attn.o_proj.")):
        key = key.replace(src, dst)
    # The native schema stores the text tower unprefixed; the CausalLM module
    # tree lives under ``model.`` (except lm_head).
    if key.startswith(("layers.", "embed_tokens.", "norm.")):
        return "model." + key
    return key


def _rank_suffix(world: int, rank: int) -> str:
    if world <= 1:
        return ""
    return f"_world{world}_rank{rank}"


def _world_suffix(world: int) -> str:
    if world <= 1:
        return ""
    return f"_world{world}"


def _is_speculative_model_config(config: "ModelConfig") -> bool:
    return any(
        bool(getattr(config, name, False)) for name in (
            "is_eagle3_draft",
            "eagle_base",
            "is_dflash_draft",
            "dflash_base",
            "is_dspark_draft",
            "dspark_base",
            "is_mtp_draft",
            "mtp_base",
            "gemma4_mtp_draft",
            "gemma4_mtp_base",
        ))


def _export_llm(model_dir: str,
                llm_out_dir: str,
                model_type: str = "",
                eagle_base: bool = False,
                eagle_draft_dir: str = "",
                fp8_embedding: bool = False,
                reduced_vocab_dir: str = "",
                mtp_base: bool = False,
                mtp_tree_base: bool = False,
                dflash_base: bool = False,
                dflash_tree_base: bool = False,
                dflash_draft_dir: str = "",
                jetspec_base: bool = False,
                jetspec_tree_base: bool = False,
                jetspec_draft_dir: str = "",
                dspark_base: bool = False,
                dspark_draft_dir: str = "",
                gemma4_mtp_base: bool = False,
                externalize_weights: "list[str] | None" = None,
                tp_size: int = 1,
                num_decoder_layers: "int | None" = None,
                skip_softmax_scale_factor: "float | None" = None,
                quantization_override: "str | None" = None) -> None:
    """Export LLM backbone via the standard tensorrt_edgellm pipeline.

    When ``tp_size > 1``, exports ``tp_size`` per-rank ONNX files named
    ``model_world{N}_rank{R}.onnx``.
    Each rank reloads the checkpoint fresh and shards weights to its
    slice on assignment.
    """
    os.makedirs(llm_out_dir, exist_ok=True)

    key_remap = None
    if model_type == "alpamayo_r1":
        key_remap = _alpamayo_llm_key_remap
    elif model_type == "cosmos3_edge":
        key_remap = _cosmos3_edge_llm_key_remap

    # ModelOpt-quantized Qwen3-MoE / Qwen3-Omni-MoE checkpoints store per-expert
    # weights under ``mlp.experts.{j}.`` (modelopt's fused-expert export, for
    # both bare Qwen3-MoE and Qwen3-Omni Thinker / Talker). The model wraps the
    # per-expert ModuleList behind a private ``_experts`` attribute, so a
    # one-segment insertion is required for the load to find the buffers.
    # Without this remap the loader silently skips every expert weight,
    # producing a Thinker engine ~3 GB (attention + norms only) instead of the
    # expected ~17 GB.
    if key_remap is None and model_type in ("qwen3_omni_moe_text",
                                            "qwen3_omni_moe_talker",
                                            "qwen3_omni_moe", "qwen3_moe"):
        from ..models.qwen3_moe import MODELOPT_KEY_REMAP
        key_remap = MODELOPT_KEY_REMAP

    # Gemma4 MoE: checkpoint stores router/experts at layer level but
    # model tree nests them under moe_block with _experts indirection.
    # Activate when the checkpoint has MoE (quantized or BF16 QAT).
    _needs_moe_quantization = False
    if key_remap is None and model_type in _GEMMA4_MODEL_TYPES:
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path) as f:
            _cfg = json.load(f)
        _llm = _cfg.get("text_config", _cfg)
        if _llm.get("enable_moe_block", False):
            _is_nvfp4 = _is_nvfp4_checkpoint(model_dir)
            _has_int4_cfg = os.path.isfile(
                os.path.join(model_dir, "hf_quant_config.json"))
            _needs_moe_quantization = (not _is_nvfp4 and not _has_int4_cfg)
            if _needs_moe_quantization:
                # BF16 QAT-unquantized: fused expert tensors
                from ..models.gemma4.modeling_gemma4_text import \
                    GEMMA4_FUSED_BF16_KEY_REMAP
                key_remap = GEMMA4_FUSED_BF16_KEY_REMAP
            else:
                # NVFP4 or INT4 AWQ: per-expert quantized weights
                from ..models.gemma4.modeling_gemma4_text import \
                    GEMMA4_NVFP4_KEY_REMAP
                key_remap = GEMMA4_NVFP4_KEY_REMAP

    ranks = [(0, 1)] if tp_size <= 1 else [(r, tp_size)
                                           for r in range(tp_size)]

    for rank, world in ranks:
        output_path = os.path.join(llm_out_dir,
                                   f"model{_rank_suffix(world, rank)}.onnx")
        if world > 1:
            logger.info("[LLM] === rank %d / %d ===", rank, world)

        logger.info("[LLM] Loading checkpoint from %s", model_dir)
        try:
            from ..model import AutoModel

            # Build config overrides for on-the-fly quantization of BF16 MoE
            # checkpoints (e.g. --quantization int4_awq on a QAT-unquantized ckpt).
            _extra_configs = None
            if quantization_override == "int4_awq" and _needs_moe_quantization:
                _extra_configs = {
                    "_needs_moe_quantization": True,
                    "_use_int4_moe_plugin": True,
                }

            model = AutoModel.from_pretrained(
                model_dir,
                device="cpu",
                eagle_base=eagle_base,
                eagle_draft_dir=eagle_draft_dir or None,
                key_remap=key_remap,
                reduced_vocab_dir=reduced_vocab_dir or None,
                mtp_base=mtp_base,
                mtp_tree_base=mtp_tree_base,
                dflash_base=dflash_base,
                dflash_tree_base=dflash_tree_base,
                dflash_draft_dir=dflash_draft_dir or None,
                jetspec_base=jetspec_base,
                jetspec_tree_base=jetspec_tree_base,
                jetspec_draft_dir=jetspec_draft_dir or None,
                dspark_base=dspark_base,
                dspark_draft_dir=dspark_draft_dir or None,
                gemma4_mtp_base=gemma4_mtp_base,
                tp_size=world,
                tp_rank=rank,
                num_decoder_layers=num_decoder_layers,
                extra_configs=_extra_configs,
            )
            if world > 1 and _is_speculative_model_config(model.config):
                raise ValueError(
                    "Tensor-parallel speculative decoding export is not supported."
                )
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            logger.exception("[LLM] Failed to load checkpoint")
            raise SystemExit(1) from exc

        # CLI override of the skip-softmax (BLASST) scale factor: None = flag not given (keep the config value); an explicit 0.0
        # disables skip-softmax even when config.json carries a positive S.
        if skip_softmax_scale_factor is not None:
            n_patched = 0
            for module in model.modules():
                if hasattr(module, "skip_softmax_scale_factor"):
                    module.skip_softmax_scale_factor = skip_softmax_scale_factor
                    n_patched += 1
            if n_patched:
                logger.info(
                    "[LLM] skip_softmax_scale_factor=%.6f applied to %d "
                    "attention modules", skip_softmax_scale_factor, n_patched)
            else:
                logger.warning(
                    "[LLM] --skip-softmax-scale-factor=%.6f had NO effect: no "
                    "attention module carries the attribute (this model's "
                    "attention variant is not wired for skip-softmax yet)",
                    skip_softmax_scale_factor)

        # Runtime config is shared at world scope; ONNX files remain rank-local.
        # Single-device exports keep the conventional "config.json".
        config_filename = f"config{_world_suffix(world)}.json"

        # Embedding and tokenizer files are shared across ranks, so write
        # them once. Every rank may rewrite the shared config; rank metadata is
        # canonicalized so common equal-split TP exports remain stable.
        logger.info("[LLM] Exporting to %s", output_path)
        try:
            from ..onnx.export import export_onnx
            export_onnx(model,
                        output_path,
                        model_dir=model_dir,
                        fp8_embedding=fp8_embedding,
                        reduced_vocab_dir=reduced_vocab_dir,
                        externalize_weights=externalize_weights,
                        config_filename=config_filename,
                        write_shared_artifacts=(rank == 0))
        except (OSError, ValueError, RuntimeError) as exc:
            logger.exception("[LLM] ONNX export failed")
            raise SystemExit(1) from exc

        # DFlash: the draft's proposal query embeds the mask token via this base
        # engine's shared embedding table, so fold the draft's trained mask row
        # into it (no-op when the row is shared with the base).
        if dflash_draft_dir and world == 1:
            from ..checkpoint.checkpoint_utils import _runtime_embedding_scale
            _patch_dflash_mask_embedding(
                llm_out_dir,
                dflash_draft_dir,
                embedding_scale=_runtime_embedding_scale(model))

        # Free this rank's model before building the next one
        del model

    # Patch multimodal token IDs into the LLM config so the C++ runtime
    # can identify which positions in the token stream must be replaced
    # with vision / audio encoder embeddings (see ``llmEngineRunner.cpp``
    # ``audio_token_id`` / ``image_token_id`` lookup).
    #
    # Source of truth: ``thinker_config`` (Qwen3-Omni) or root config
    # (Qwen-VL / Nemotron-Omni).  Naming conventions vary across checkpoints
    # — see :func:`_patch_multimodal_token_ids` for the fallback chain.
    _patch_multimodal_token_ids(model_dir,
                                llm_out_dir,
                                model_type,
                                config_filename=config_filename)

    # Standalone Talker checkpoints route through ``_export_llm`` (not the
    # qwen3_tts ``_export_talker``) because their model_type isn't in the
    # ``_LLM_COMPONENTS`` Talker-dispatch map. They still need the TTS
    # config fields (accept_hidden_layer, speaker_id, default_speaker_id,
    # tts_*_token_id, codec_*_id) that runtime ``Qwen3OmniTTSRuntime``
    # validates. Without these the engine config has nulls and the C++
    # runtime falls back to wrong defaults → Talker doesn't EOS properly.
    if model_type in ("qwen3_omni_moe_talker", "qwen3_omni_talker"):
        _patch_tts_config(model_dir, llm_out_dir)

    logger.info("[LLM] Done: %s", output_path)


def _patch_diffusion_export_config(model_dir: str, output_dir: str) -> None:
    """Patch DiffusionGemma runtime metadata that is not present in HF config.json."""
    root = _load_config(model_dir)
    token_ids = {}
    for key in ("image_token_id", "audio_token_id"):
        value = root.get(key)
        if isinstance(value, int):
            token_ids[key] = value
    if "image_token_id" not in token_ids:
        image_token_id = _find_token_id(model_dir, "<|image_pad|>")
        if image_token_id is not None:
            token_ids["image_token_id"] = image_token_id
    if not token_ids:
        return

    cfg_patch_path = os.path.join(output_dir, "dllm", "config.json")
    if not os.path.exists(cfg_patch_path):
        return
    with open(cfg_patch_path) as f:
        cfg = json.load(f)
    cfg.update(token_ids)
    with open(cfg_patch_path, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("[DiffusionGemma] Patched multimodal token IDs into %s",
                cfg_patch_path)


def _diffusion_gemma_has_visual(config: dict) -> bool:
    return isinstance(config.get("vision_config"), dict)


def _remap_diffusion_gemma_visual_weights(weights: dict) -> dict:
    """Return a Gemma4-visual-compatible view of DiffusionGemma visual weights."""
    remapped = {}
    prefix_map = (
        ("model.encoder.vision_tower.", "model.vision_tower."),
        ("model.encoder.embed_vision.", "model.embed_vision."),
        ("model.encoder.multimodal_embedder.", "model.embed_vision."),
        ("model.encoder.mm_soft_embedding_projection.",
         "model.embed_vision.embedding_projection."),
        ("model.encoder.mm_soft_embedding_norm.",
         "model.embed_vision.embedding_pre_projection_norm."),
    )
    for key, tensor in weights.items():
        for src, dst in prefix_map:
            if key.startswith(src):
                remapped[dst + key[len(src):]] = tensor
                break
    if not remapped:
        raise ValueError("DiffusionGemma checkpoint has vision_config but no "
                         "model.encoder visual weights were found.")
    return remapped


def _export_diffusion_gemma_visual(model_dir: str, visual_out_dir: str,
                                   weights: dict, config: dict,
                                   dtype: "torch.dtype",
                                   model_config: "ModelConfig") -> None:
    """Export DiffusionGemma vision tower via the Gemma4 visual exporter."""
    if not _diffusion_gemma_has_visual(config):
        raise ValueError("DiffusionGemma checkpoint has no vision_config.")
    visual_weights = _remap_diffusion_gemma_visual_weights(weights)
    _export_visual(model_dir,
                   visual_out_dir,
                   visual_weights,
                   config,
                   "gemma4",
                   dtype,
                   model_config=model_config)


def _export_diffusion_gemma(model_dir: str,
                            output_dir: str,
                            fp8_embedding: bool = False,
                            reduced_vocab_dir: str = "",
                            externalize_weights: "list[str] | None" = None,
                            tp_size: int = 1) -> None:
    """Export DiffusionGemma as one unified backbone graph."""
    if tp_size != 1:
        raise SystemExit(
            "DiffusionGemma export currently supports only --tp-size 1.")

    backbone_out_dir = os.path.join(output_dir, "dllm")
    os.makedirs(backbone_out_dir, exist_ok=True)
    backbone_externalize_weights = _diffusion_gemma_backbone_externalize_weights(
        model_dir, externalize_weights)
    use_nvfp4_moe = _is_nvfp4_checkpoint(model_dir)

    try:
        from ..model import AutoModel
        from ..models.diffusion_gemma import make_diffusion_gemma_key_remap
        from ..models.linear import FP16Linear
        from ..onnx.export import export_onnx
    except ImportError as exc:
        logger.exception("[DiffusionGemma] Failed to import export helpers")
        raise SystemExit(1) from exc

    logger.info("[DiffusionGemma] Loading backbone checkpoint from %s",
                model_dir)
    try:
        backbone = AutoModel.from_pretrained(
            model_dir,
            device="cpu",
            key_remap=make_diffusion_gemma_key_remap(
                include_backbone=True,
                include_self_conditioning=True,
                nvfp4_moe=use_nvfp4_moe),
            reduced_vocab_dir=reduced_vocab_dir or None,
        )
        for name in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(backbone.self_conditioning, name)
            if not isinstance(module, FP16Linear):
                raise ValueError(
                    "DiffusionGemma unified self-conditioning requires "
                    f"self_conditioning.{name} to export as FP16Linear; "
                    f"got {type(module).__name__}. Add this module to "
                    "the checkpoint quantization exclude list.")
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.exception("[DiffusionGemma] Failed to load backbone")
        raise SystemExit(1) from exc

    backbone_path = os.path.join(backbone_out_dir, "model.onnx")
    logger.info("[DiffusionGemma] Exporting backbone to %s", backbone_path)
    try:
        export_onnx(backbone,
                    backbone_path,
                    model_dir=model_dir,
                    fp8_embedding=fp8_embedding,
                    reduced_vocab_dir=reduced_vocab_dir,
                    externalize_weights=backbone_externalize_weights)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[DiffusionGemma] Backbone ONNX export failed")
        raise SystemExit(1) from exc

    del backbone

    _patch_diffusion_export_config(model_dir, output_dir)
    logger.info("[DiffusionGemma] Done: %s", output_dir)


def _export_mtp_draft(model_dir: str,
                      draft_out_dir: str,
                      externalize_weights: "list[str] | None" = None) -> None:
    """Export the MTP draft model."""
    os.makedirs(draft_out_dir, exist_ok=True)
    output_path = os.path.join(draft_out_dir, "model.onnx")

    logger.info("[MTP Draft] Loading checkpoint from %s", model_dir)
    try:
        from ..model import AutoModel
        model = AutoModel.from_pretrained(model_dir,
                                          device="cpu",
                                          mtp_draft=True)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.exception("[MTP Draft] Failed to load checkpoint")
        raise SystemExit(1) from exc

    logger.info("[MTP Draft] Exporting to %s", output_path)
    try:
        from ..onnx.export import export_onnx
        export_onnx(model,
                    output_path,
                    model_dir=model_dir,
                    externalize_weights=externalize_weights)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[MTP Draft] ONNX export failed")
        raise SystemExit(1) from exc

    logger.info("[MTP Draft] Done: %s", output_path)


def _export_gemma4_mtp_draft(target_dir: str, draft_out_dir: str,
                             assistant_dir: str,
                             kv_sharing_map: list[dict]) -> None:
    """Export the paired Gemma4 assistant draft model."""
    os.makedirs(draft_out_dir, exist_ok=True)
    output_path = os.path.join(draft_out_dir, "model.onnx")
    logger.info("[Gemma4 MTP Draft] Target checkpoint: %s", target_dir)
    logger.info("[Gemma4 MTP Draft] Loading assistant checkpoint from %s",
                assistant_dir)
    try:
        from ..checkpoint.checkpoint_utils import write_runtime_artifacts
        from ..model import AutoModel, load_model_config
        target_model_config = load_model_config(target_dir)
        model = AutoModel.from_pretrained(
            assistant_dir,
            device="cpu",
            gemma4_mtp_draft=True,
            gemma4_kv_sharing_map=kv_sharing_map,
            gemma4_target_kv_cache_quant=target_model_config.quant.
            kv_cache_quant,
        )
        write_runtime_artifacts(model, assistant_dir, draft_out_dir)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.exception("[Gemma4 MTP Draft] Failed to load assistant")
        raise SystemExit(1) from exc

    logger.info("[Gemma4 MTP Draft] Exporting assistant ONNX to %s",
                output_path)
    try:
        from ..onnx.export import export_onnx
        export_onnx(model, output_path, model_dir=assistant_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[Gemma4 MTP Draft] ONNX export failed")
        raise SystemExit(1) from exc

    logger.info("[Gemma4 MTP Draft] Done: %s", output_path)


def _export_dflash_draft(model_dir: str,
                         draft_out_dir: str,
                         dflash_draft_dir: str,
                         draft_reduced_vocab_dir: str = "") -> None:
    """Export the DFlash draft model, optionally with reduced vocabulary."""
    os.makedirs(draft_out_dir, exist_ok=True)
    output_path = os.path.join(draft_out_dir, "model.onnx")

    logger.info("[DFlash Draft] Loading checkpoint from %s", dflash_draft_dir)
    try:
        from ..model import AutoModel
        model = AutoModel.from_pretrained(model_dir,
                                          device="cpu",
                                          dflash_draft=True,
                                          dflash_draft_dir=dflash_draft_dir)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.exception("[DFlash Draft] Failed to load checkpoint")
        raise SystemExit(1) from exc

    # --- Optional: reduce draft lm_head vocabulary ---
    full_size = model.config.vocab_size
    reduced_size = None
    if draft_reduced_vocab_dir:
        logger.info("[DFlash Draft] Applying vocab reduction from %s",
                    draft_reduced_vocab_dir)
        try:
            from ..vocab_reduction.onnx_export import \
                apply_reduced_vocab_from_dir
            apply_reduced_vocab_from_dir(model, draft_reduced_vocab_dir)
            reduced_size = model.config.reduced_vocab_size
            logger.info("[DFlash Draft] lm_head reduced: %d → %d", full_size,
                        reduced_size)
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            logger.exception("[DFlash Draft] Vocab reduction failed")
            raise SystemExit(1) from exc

    logger.info("[DFlash Draft] Exporting to %s", output_path)
    try:
        from ..onnx.export import export_onnx
        export_onnx(model, output_path, model_dir=dflash_draft_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[DFlash Draft] ONNX export failed")
        raise SystemExit(1) from exc

    # FP16/FP32 RoPE fix is handled automatically by export_onnx() which
    # reads DFlashDraftModel.match_fp32_elementwise_initializers = True
    # and passes it to _fix_initializer_dtypes().

    # --- Save draft vocab map sidecar for C++ runtime ---
    if draft_reduced_vocab_dir:
        from tensorrt_edgellm._safetensors_io import \
            save_file as _save_safetensors

        from ..vocab_reduction.constants import (DRAFT_VOCAB_INFO_NAME,
                                                 DRAFT_VOCAB_MAP_NAME)
        vocab_map = model._reduced_vocab_map_for_runtime

        map_path = os.path.join(draft_out_dir, DRAFT_VOCAB_MAP_NAME)
        _save_safetensors({"vocab_map": vocab_map.cpu().to(torch.int32)},
                          map_path)
        logger.info("[DFlash Draft] Wrote draft vocab map: %s (%d tokens)",
                    map_path, vocab_map.numel())

        with open(os.path.join(draft_out_dir, DRAFT_VOCAB_INFO_NAME),
                  "w") as fh:
            json.dump(
                {
                    "vocab_size": full_size,
                    "reduced_vocab_size": reduced_size,
                    "source": draft_reduced_vocab_dir
                },
                fh,
                indent=2)

    logger.info("[DFlash Draft] Done: %s", output_path)


def _export_jetspec_draft(model_dir: str,
                          draft_out_dir: str,
                          jetspec_draft_dir: str,
                          draft_reduced_vocab_dir: str = "") -> None:
    """Export the JetSpec draft model, optionally with reduced vocabulary."""
    os.makedirs(draft_out_dir, exist_ok=True)
    output_path = os.path.join(draft_out_dir, "model.onnx")

    logger.info("[JetSpec Draft] Loading checkpoint from %s",
                jetspec_draft_dir)
    try:
        from ..model import AutoModel
        model = AutoModel.from_pretrained(model_dir,
                                          device="cpu",
                                          jetspec_draft=True,
                                          jetspec_draft_dir=jetspec_draft_dir)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.exception("[JetSpec Draft] Failed to load checkpoint")
        raise SystemExit(1) from exc

    full_size = model.config.vocab_size
    reduced_size = None
    if draft_reduced_vocab_dir:
        logger.info("[JetSpec Draft] Applying vocab reduction from %s",
                    draft_reduced_vocab_dir)
        try:
            from ..vocab_reduction.onnx_export import \
                apply_reduced_vocab_from_dir
            apply_reduced_vocab_from_dir(model, draft_reduced_vocab_dir)
            reduced_size = model.config.reduced_vocab_size
            logger.info("[JetSpec Draft] lm_head reduced: %d -> %d", full_size,
                        reduced_size)
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            logger.exception("[JetSpec Draft] Vocab reduction failed")
            raise SystemExit(1) from exc

    logger.info("[JetSpec Draft] Exporting to %s", output_path)
    try:
        from ..onnx.export import export_onnx
        export_onnx(model, output_path, model_dir=jetspec_draft_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[JetSpec Draft] ONNX export failed")
        raise SystemExit(1) from exc

    if draft_reduced_vocab_dir:
        from tensorrt_edgellm._safetensors_io import \
            save_file as _save_safetensors

        from ..vocab_reduction.constants import (DRAFT_VOCAB_INFO_NAME,
                                                 DRAFT_VOCAB_MAP_NAME)
        vocab_map = model._reduced_vocab_map_for_runtime
        map_path = os.path.join(draft_out_dir, DRAFT_VOCAB_MAP_NAME)
        _save_safetensors({"vocab_map": vocab_map.cpu().to(torch.int32)},
                          map_path)
        logger.info("[JetSpec Draft] Wrote draft vocab map: %s (%d tokens)",
                    map_path, vocab_map.numel())
        with open(os.path.join(draft_out_dir, DRAFT_VOCAB_INFO_NAME),
                  "w") as fh:
            json.dump(
                {
                    "vocab_size": full_size,
                    "reduced_vocab_size": reduced_size,
                    "source": draft_reduced_vocab_dir
                },
                fh,
                indent=2)

    logger.info("[JetSpec Draft] Done: %s", output_path)


def _patch_dflash_mask_embedding(llm_out_dir: str,
                                 dflash_draft_dir: str,
                                 embedding_scale: float = 1.0) -> None:
    """Fold the DFlash draft's trained mask-token embedding into the base sidecar.

    The runtime embeds the draft proposal query ``[anchor, mask, ...]`` by
    looking ``mask_token_id`` up in the base engine's shared
    ``embedding.safetensors`` (``dflashDecoder.cpp`` ``runDraftForward``). Some
    DFlash checkpoints (e.g. Nemotron-3.5) ship a distinct trained embedding for
    that reserved token in the draft's own ``embed_tokens`` — typically the only
    row that differs from the base table. Patch only that single row; when the
    draft's mask row is shared with the base (the common Qwen-style case) this is
    a no-op.
    """
    import glob

    from safetensors import safe_open

    from tensorrt_edgellm._safetensors_io import save_file

    emb_path = os.path.join(llm_out_dir, "embedding.safetensors")
    if not os.path.exists(emb_path):
        logger.warning("[DFlash] %s missing; cannot fold draft mask embedding",
                       emb_path)
        return

    draft_cfg = _load_config(dflash_draft_dir)
    dcfg = draft_cfg.get("dflash_config", {}) or {}
    mask_id = dcfg.get("mask_token_id", draft_cfg.get("mask_token_id"))
    if mask_id is None:
        logger.warning("[DFlash] draft config has no mask_token_id; skipping "
                       "mask embedding fold")
        return
    mask_id = int(mask_id)

    # embed_tokens is excluded from draft quantization, so the row is dense.
    draft_vec = None
    for shard in sorted(
            glob.glob(os.path.join(dflash_draft_dir, "*.safetensors"))):
        with safe_open(shard, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for key in ("embed_tokens.weight", "model.embed_tokens.weight"):
                if key in keys:
                    draft_vec = f.get_slice(key)[mask_id:mask_id +
                                                 1].squeeze(0).to(
                                                     torch.float32)
                    break
        if draft_vec is not None:
            break
    if draft_vec is None:
        logger.info("[DFlash] draft checkpoint has no embed_tokens; mask "
                    "embedding is shared with the base (no fold needed)")
        return

    with safe_open(emb_path, framework="pt", device="cpu") as f:
        if "embedding_scale" in set(f.keys()):
            raise ValueError(
                "DFlash mask-embedding fold does not support FP8 "
                "embedding.safetensors; re-export the base without "
                "--fp8-embedding.")
        weight = f.get_tensor("embedding")

    patched_row = (draft_vec * embedding_scale).to(weight.dtype)
    if torch.allclose(weight[mask_id].to(torch.float32),
                      patched_row.to(torch.float32),
                      atol=1e-3,
                      rtol=0.0):
        logger.info(
            "[DFlash] draft mask embedding (id=%d) matches base; no fold needed",
            mask_id)
        return

    weight[mask_id] = patched_row
    # Write to a temp file and atomically rename so an interrupted patch never
    # leaves the base engine's embedding sidecar half-written; a re-run then
    # recovers without re-exporting the base.
    tmp_path = emb_path + ".tmp"
    save_file({"embedding": weight.contiguous()}, tmp_path)
    os.replace(tmp_path, emb_path)
    logger.info(
        "[DFlash] Folded draft mask embedding (id=%d) into base "
        "embedding.safetensors", mask_id)


_DSPARK_HEAD_TENSOR_KEYS = {
    "markov_w1": "markov_head.markov_w1.weight",
    "markov_w2": "markov_head.markov_w2.weight",
    "confidence_weight": "confidence_head.proj.weight",
    "confidence_bias": "confidence_head.proj.bias",
}


def _load_dspark_head_tensors(draft_dir: str, required_keys: set[str]) -> dict:
    """Load DSpark Markov/confidence sidecar tensors from safetensors shards."""
    import glob

    from safetensors import safe_open

    shards = sorted(glob.glob(os.path.join(draft_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No safetensors files found in {draft_dir}")

    remaining = dict(_DSPARK_HEAD_TENSOR_KEYS)
    loaded: dict = {}
    source: dict = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for save_name, ckpt_key in list(remaining.items()):
                if ckpt_key in keys:
                    tensor = f.get_tensor(ckpt_key).cpu()
                    if save_name == "confidence_weight" and tensor.ndim == 2 and tensor.shape[
                            0] == 1:
                        tensor = tensor.squeeze(0)
                    loaded[save_name] = tensor
                    source[save_name] = {
                        "checkpoint_key": ckpt_key,
                        "shard": os.path.basename(shard),
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype).replace("torch.", ""),
                    }
                    del remaining[save_name]

    missing_required = sorted(required_keys - loaded.keys())
    if missing_required:
        details = ", ".join(f"{name}<-{_DSPARK_HEAD_TENSOR_KEYS[name]}"
                            for name in missing_required)
        raise KeyError(
            f"DSpark draft checkpoint is missing required head tensors: {details}"
        )
    return loaded, source


def _export_dspark_sidecars(dspark_draft_dir: str, draft_out_dir: str) -> None:
    """Export DSpark Markov and confidence heads as runtime sidecars.

    The draft ONNX engine contains the parallel DSpark backbone only. The
    sequential Markov head and confidence scheduler are intentionally kept as
    sidecar tensors so the runtime can execute token-by-token scheduling
    without inflating the TensorRT graph with dynamic control flow.
    """
    from tensorrt_edgellm._safetensors_io import save_file

    cfg = _load_config(dspark_draft_dir)
    dspark_cfg = cfg.get("dspark_config", {}) or {}
    markov_type = str(
        dspark_cfg.get("markov_head_type", cfg.get("markov_head_type", "")))
    enable_confidence = bool(
        dspark_cfg.get("enable_confidence_head",
                       cfg.get("enable_confidence_head", False)))
    confidence_with_markov = bool(
        dspark_cfg.get("confidence_head_with_markov",
                       cfg.get("confidence_head_with_markov", False)))
    required = {"markov_w1", "markov_w2"}
    if enable_confidence:
        required.update({"confidence_weight", "confidence_bias"})

    tensors, source = _load_dspark_head_tensors(dspark_draft_dir, required)
    out_tensors = _to_fp16(tensors)
    heads_path = os.path.join(draft_out_dir, "dspark_heads.safetensors")
    save_file(out_tensors, heads_path)

    info = {
        "format":
        "tensorrt-edgellm-dspark-heads-v1",
        "source":
        dspark_draft_dir,
        "markov_head_type":
        markov_type,
        "markov_rank":
        int(dspark_cfg.get("markov_rank", cfg.get("markov_rank", 0)) or 0),
        "enable_confidence_head":
        enable_confidence,
        "confidence_head_with_markov":
        confidence_with_markov,
        "tensor_keys":
        sorted(out_tensors.keys()),
        "source_tensors":
        source,
        "notes": [
            "markov_w2 is saved in checkpoint Linear weight layout [vocab_size, markov_rank].",
            "confidence_weight is squeezed to [input_dim] when the checkpoint stores [1, input_dim].",
        ],
    }
    info_path = os.path.join(draft_out_dir, "dspark_heads_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info("[DSpark Draft] Wrote head sidecars: %s, %s", heads_path,
                info_path)


def _export_dspark_draft(model_dir: str, draft_out_dir: str,
                         dspark_draft_dir: str) -> None:
    """Export the DSpark draft backbone plus Markov/confidence sidecars."""
    os.makedirs(draft_out_dir, exist_ok=True)
    output_path = os.path.join(draft_out_dir, "model.onnx")

    logger.info("[DSpark Draft] Loading checkpoint from %s", dspark_draft_dir)
    try:
        from ..model import AutoModel
        model = AutoModel.from_pretrained(model_dir,
                                          device="cpu",
                                          dspark_draft=True,
                                          dspark_draft_dir=dspark_draft_dir)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.exception("[DSpark Draft] Failed to load checkpoint")
        raise SystemExit(1) from exc

    logger.info("[DSpark Draft] Exporting backbone ONNX to %s", output_path)
    try:
        from ..onnx.export import export_onnx
        export_onnx(model, output_path, model_dir=dspark_draft_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[DSpark Draft] ONNX export failed")
        raise SystemExit(1) from exc

    try:
        _export_dspark_sidecars(dspark_draft_dir, draft_out_dir)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        logger.exception("[DSpark Draft] Sidecar export failed")
        raise SystemExit(1) from exc

    logger.info("[DSpark Draft] Done: %s", output_path)


def _tower_model_config(model_config: "ModelConfig", weights: dict,
                        prefixes: tuple) -> "ModelConfig":
    """Return *model_config* with quant reset to fp16 when the tower under
    *prefixes* ships no ``.weight_scale`` tensors (i.e. it was left
    unquantized in an otherwise-quantized consolidated checkpoint).

    Detects per-tower quantization from the checkpoint instead of assuming
    the backbone quant_type applies (same pattern as the CodePredictor).
    """
    import dataclasses

    from ..config import QuantConfig
    tower_quantized = any(
        k.endswith(".weight_scale") and k.startswith(prefixes)
        for k in weights)
    if tower_quantized:
        return model_config
    return dataclasses.replace(model_config, quant=QuantConfig())


def _export_visual(model_dir: str, visual_out_dir: str, weights: dict,
                   config: dict, model_type: str, dtype: "torch.dtype",
                   model_config: "ModelConfig") -> None:
    """Export visual encoder via from-scratch tensorrt_edgellm pipeline.

    For Qwen3-Omni-specific checkpoint-key translation (``thinker.visual.*``
    to ``model.visual.*`` plus merger sub-module renames), see
    :mod:`tensorrt_edgellm.models.qwen3_omni.modeling_qwen3_omni_visual`. That
    family is registered in ``_VISUAL_REGISTRY`` for
    ``qwen3_omni`` / ``qwen3_omni_moe`` model_types and runs the
    remap inside its own ``build_qwen3_omni_visual``.
    """
    from ..quantization.quantization_configs import _VISUAL_PREFIXES
    tower_prefixes = tuple(f"{root}{p}." for p in _VISUAL_PREFIXES
                           for root in ("", "model.", "thinker.",
                                        "model.embed_tokens_extend."))
    model_config = _tower_model_config(model_config, weights, tower_prefixes)
    os.makedirs(visual_out_dir, exist_ok=True)
    output_path = os.path.join(visual_out_dir, "model.onnx")

    logger.info("[Visual] Exporting %s visual encoder to %s", model_type,
                output_path)
    try:
        from ..onnx.export_encoder import export_visual_onnx
        export_visual_onnx(
            model_dir=model_dir,
            output_path=output_path,
            weights=weights,
            config=config,
            model_type=model_type,
            dtype=dtype,
            model_config=model_config,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[Visual] ONNX export failed")
        raise SystemExit(1) from exc
    logger.info("[Visual] Done: %s", output_path)

    # Write a config.json for the C++ runtime.
    # The visual builder will merge builder_config into this file when it
    # builds the engine. Fields needed vary by model family:
    #   InternVL*: image_token_id, text_config (for vocab_size)
    #   all:       model_type, vision_config
    # Qwen3-Omni nests vision_config / text_config / token IDs under
    # thinker_config; other Qwen VL variants keep them at the root.
    _thinker_cfg = config.get("thinker_config", {}) or {}
    vis_cfg = (config.get("vision_config") or _thinker_cfg.get("vision_config")
               or config)
    # Map the top-level checkpoint model_type to the encoder-specific
    # enum the C++ ``stringToModelType`` expects for the visual builder.
    # - ``internvl`` / ``internvl_chat`` → ``internvl``
    # - ``qwen3_omni`` → ``qwen3_omni_vision_encoder`` (bare "qwen3_omni"
    #   maps to AUDIO_ENCODER in C++, so visualBuilder rejects it)
    _VISUAL_MODEL_TYPE_MAP = {
        "internvl": "internvl",
        "internvl_chat": "internvl",
        "qwen3_5_moe": "qwen3_5",
        "qwen3_omni": "qwen3_omni_vision_encoder",
        # MoE variant uses byte-identical visual encoder weights; reuse the
        # same C++ runner enum the dense Qwen3-Omni visual engine registers.
        "qwen3_omni_moe": "qwen3_omni_vision_encoder",
        "gemma4": "gemma4_vision",
        "qwen3_omni_next": "qwen3_omni_next_vision_encoder",
        "gemma4_unified": "gemma4_unified_vision",
        # Cosmos3-Edge reasoner SigLIP2 ViT (the bare "cosmos3_edge" maps to
        # the text decoder in C++; the visual engine registers its own enum).
        "cosmos3_edge": "cosmos3_edge_vision",
    }
    top_level_model_type = _VISUAL_MODEL_TYPE_MAP.get(model_type, model_type)
    vis_cfg_out: dict = {
        "model_type": top_level_model_type,
        "vision_config": vis_cfg,
    }
    if model_type == "qwen3_5_moe":
        # The C++ visual builder prefers vision_config.model_type over the
        # top-level model_type.  Qwen3.5-MoE uses the same visual encoder as
        # dense Qwen3.5, so normalize both locations to the registered tag.
        vis_cfg_out["vision_config"] = dict(vis_cfg_out["vision_config"])
        vis_cfg_out["vision_config"]["model_type"] = "qwen3_5"
    if model_type == "qwen3_omni_moe":
        # Same reason as qwen3_5_moe: nested ``vision_config.model_type``
        # in HF Qwen3-Omni-MoE is ``qwen3_omni_moe_vision_encoder`` which
        # the C++ visualBuilder enum doesn't recognise. Reuse the
        # ``qwen3_omni_vision_encoder`` registration since the visual
        # encoder is byte-identical between dense and MoE Qwen3-Omni.
        vis_cfg_out["vision_config"] = dict(vis_cfg_out["vision_config"])
        vis_cfg_out["vision_config"][
            "model_type"] = "qwen3_omni_vision_encoder"
    if model_type in ("qwen2_5_vl", "qwen3_vl", "qwen3_omni", "qwen3_omni_moe",
                      "qwen3_omni_next", "qwen3_5", "qwen3_5_moe",
                      "cosmos3_edge"):
        # C++ QwenViTRunner reads these token IDs and rope_theta from config.json.
        # For Qwen3-VL the token IDs are at the root level, but vocab_size and
        # rope_theta live inside text_config.  Fall back to text_config for any
        # key that is absent from the root.  For Qwen3-Omni all of these live
        # under thinker_config (token IDs) and thinker_config.text_config.
        _text_cfg = (config.get("text_config")
                     or _thinker_cfg.get("text_config") or {})
        # rope_theta may live in text_config.rope_parameters (newer transformers)
        _rope_params = _text_cfg.get("rope_parameters") or _text_cfg.get(
            "rope_scaling") or {}
        for key in ("vision_start_token_id", "vision_end_token_id",
                    "image_token_id", "video_token_id", "vocab_size",
                    "rope_theta"):
            if key in config:
                vis_cfg_out[key] = config[key]
            elif key in _thinker_cfg:
                vis_cfg_out[key] = _thinker_cfg[key]
            elif key in _text_cfg:
                vis_cfg_out[key] = _text_cfg[key]
            elif key in _rope_params:
                vis_cfg_out[key] = _rope_params[key]
        # Include rope_scaling (contains mrope_section) for Qwen VL models.
        # The C++ QwenViTRunner reads mrope_section from rope_scaling.
        # Quantized checkpoints may use rope_parameters instead of
        # rope_scaling — normalize to rope_scaling for the C++ runtime.
        _rope_scaling = (_text_cfg.get("rope_scaling")
                         or _text_cfg.get("rope_parameters")
                         or config.get("rope_scaling")
                         or config.get("rope_parameters"))
        if _rope_scaling:
            vis_cfg_out["rope_scaling"] = normalize_rope_scaling_for_runtime(
                _rope_scaling)
    if model_type in ("gemma4", "gemma4_unified"):
        vis_cfg_out["vision_config"] = dict(vis_cfg_out["vision_config"])
        visual_model_type = ("gemma4_unified_vision" if model_type
                             == "gemma4_unified" else "gemma4_vision")
        vis_cfg_out["model_type"] = visual_model_type
        vis_cfg_out["vision_config"]["model_type"] = visual_model_type
        text_cfg = config.get("text_config") or {}
        if text_cfg:
            vis_cfg_out["text_config"] = text_cfg
        for key in ("image_token_id", "audio_token_id", "boi_token_id",
                    "eoi_token_id", "boa_token_id", "eoa_token_index"):
            if key in config:
                vis_cfg_out[key] = config[key]
        if "image_token_id" not in vis_cfg_out:
            image_token = ("<|image|>" if model_type == "gemma4_unified" else
                           "<|image_pad|>")
            image_token_id = _find_token_id(model_dir, image_token)
            if image_token_id is not None:
                vis_cfg_out["image_token_id"] = image_token_id
        if "audio_token_id" not in vis_cfg_out:
            audio_token = ("<|audio|>" if model_type == "gemma4_unified" else
                           "<|audio_pad|>")
            audio_token_id = _find_token_id(model_dir, audio_token)
            if audio_token_id is not None:
                vis_cfg_out["audio_token_id"] = audio_token_id
        if model_type == "gemma4_unified":
            # Unified images have a fixed upper bound on the number of soft
            # tokens produced for each image.  Persist that bound in the
            # exporter sidecar so both visualBuilder's optimization profile
            # and the runtime runner agree before the engine config is
            # generated.
            max_soft_tokens = vis_cfg_out["vision_config"].get(
                "num_soft_tokens")
            processor_path = os.path.join(model_dir, "processor_config.json")
            if os.path.exists(processor_path):
                with open(processor_path) as f:
                    processor_config = json.load(f)
                image_processor = processor_config.get("image_processor") or {}
                max_soft_tokens = image_processor.get("max_soft_tokens",
                                                      max_soft_tokens)
            if max_soft_tokens is not None:
                max_soft_tokens = int(max_soft_tokens)
                vis_cfg_out["builder_config"] = {
                    "max_image_tokens_per_image": max_soft_tokens,
                }
    if model_type == "qwen3_omni_moe":
        # HF Qwen3-Omni-MoE 30B-A3B-Instruct vision_config omits the
        # ``num_position_embeddings`` field that QwenViTRunner reads, but
        # ships ``image_size`` + ``patch_size`` from which it is unambiguously
        # derivable. Inject the derived value into the on-disk config so the
        # C++ runtime constructor (which reads from the engine config rather
        # than re-deriving) does not throw a silent ``out_of_range`` exception
        # during visual-runner load.
        vc_out = vis_cfg_out.get("vision_config", {})
        if isinstance(vc_out,
                      dict) and "num_position_embeddings" not in vc_out:
            _img = vc_out.get("image_size")
            _pat = vc_out.get("patch_size")
            if _img is not None and _pat:
                _grid = int(_img) // int(_pat)
                vc_out = dict(vc_out)
                vc_out["num_position_embeddings"] = _grid * _grid
                vis_cfg_out["vision_config"] = vc_out
    if model_type in ("qwen3_omni", "qwen3_omni_moe", "qwen3_omni_next"):
        # Qwen3OmniViTRunner reads position_id_per_seconds from vision_config;
        # HF stores it one level up. Copy it in.
        _pips = _thinker_cfg.get("position_id_per_seconds",
                                 config.get("position_id_per_seconds"))
        if _pips is not None and isinstance(vis_cfg_out.get("vision_config"),
                                            dict):
            vc_out = dict(vis_cfg_out["vision_config"])
            vc_out["position_id_per_seconds"] = _pips
            vis_cfg_out["vision_config"] = vc_out
    # Copy preprocessor_config.json to the visual output dir so the C++
    # runtime can find patch_size, image_mean, image_std, etc.  Applies to
    # every visual family (Qwen VL, InternVL, Phi-4mm) — the C++ visual
    # runners all read from this file.
    import shutil
    proc_src = os.path.join(model_dir, "processor_config.json")
    pp_src = os.path.join(model_dir, "preprocessor_config.json")
    if os.path.exists(pp_src):
        shutil.copy2(pp_src, visual_out_dir)
        logger.info("[Visual] Copied preprocessor_config.json to %s",
                    visual_out_dir)
    else:
        # Newer quantized checkpoints store image processor config inside
        # processor_config.json under the "image_processor" key.  Extract
        # it and write a standalone preprocessor_config.json.
        if os.path.exists(proc_src):
            with open(proc_src) as _pf:
                proc_cfg = json.load(_pf)
            img_proc = proc_cfg.get("image_processor", {})
            if img_proc:
                pp_dst = os.path.join(visual_out_dir,
                                      "preprocessor_config.json")
                with open(pp_dst, "w") as _pf:
                    json.dump(img_proc, _pf, indent=2)
                logger.info(
                    "[Visual] Extracted preprocessor_config.json from "
                    "processor_config.json to %s", visual_out_dir)
            else:
                logger.warning(
                    "[Visual] processor_config.json has no "
                    "image_processor key at %s", proc_src)
        else:
            logger.warning(
                "[Visual] Neither preprocessor_config.json nor "
                "processor_config.json found at %s", model_dir)
    if model_type in ("phi4mm", "phi4_multimodal"):
        # C++ Phi4MMViTRunner reads vocab_size and embd_layer from the top level
        # of config.json.  For phi4mm the raw config.json is flat (no vision_config
        # sub-key), so flatten the required fields from vision_config → top level.
        vc = vis_cfg_out.get("vision_config", {})
        for key in ("vocab_size", "embd_layer"):
            if key in vc:
                vis_cfg_out[key] = vc[key]
    if model_type in ("internvl", "internvl_chat"):
        # C++ internViTRunner reads image_token_id and text_config.vocab_size.
        # For internvl_chat the image token is <IMG_CONTEXT>; find it from the
        # tokenizer added_tokens list if it's not in config.json directly.
        image_token_id = config.get("image_token_id")
        if image_token_id is None:
            image_token_id = _find_token_id(model_dir, "<IMG_CONTEXT>")
        if image_token_id is not None:
            vis_cfg_out["image_token_id"] = image_token_id
        # text_config (vocab_size) — internvl_chat may use llm_config instead
        text_cfg = config.get("text_config") or config.get("llm_config")
        if text_cfg:
            vis_cfg_out["text_config"] = text_cfg
        # The C++ visual builder reads vision_config.model_type first.
        # intern_vit_6b (old arch) is not registered; override to "internvl".
        if "vision_config" in vis_cfg_out and "model_type" in vis_cfg_out[
                "vision_config"]:
            vis_cfg_out["vision_config"] = dict(vis_cfg_out["vision_config"])
            vis_cfg_out["vision_config"]["model_type"] = "internvl"
        # C++ builder reads patch_size[0]/[1] and image_size[0]/[1] as arrays.
        # Convert scalar ints to [H, W] pairs if needed.
        vc_out = vis_cfg_out["vision_config"]
        if isinstance(vc_out.get("patch_size"), int):
            vis_cfg_out["vision_config"] = dict(vc_out)
            vis_cfg_out["vision_config"]["patch_size"] = [
                vc_out["patch_size"], vc_out["patch_size"]
            ]
        if isinstance(vc_out.get("image_size"), int):
            vis_cfg_out["vision_config"] = dict(vis_cfg_out["vision_config"])
            vis_cfg_out["vision_config"]["image_size"] = [
                vc_out["image_size"], vc_out["image_size"]
            ]
    if model_type in _NEMOTRON_OMNI_MODEL_TYPES:
        # The ckpt's "NemotronH_Nano_VL_V2" is not registered in C++
        # stringToModelType(), and newer Nemotron-Omni checkpoints use
        # "NemotronH_Nano_Omni_Reasoning_V3".  Override both to the registered
        # runtime tag at top level (read by MultimodalRunner::create) and under
        # vision_config (preferred by visualBuilder).
        vis_cfg_out["model_type"] = "nemotron_omni_vision_encoder"
        vis_cfg_out["vision_config"] = dict(vis_cfg_out["vision_config"])
        vis_cfg_out["vision_config"][
            "model_type"] = "nemotron_omni_vision_encoder"
        # NemotronOmniViTRunner reads these top-level fields; visualBuilder
        # additionally reads patch_size, downsample_ratio and vit_hidden_size.
        for key in ("llm_config", "img_context_token_id", "img_start_token_id",
                    "img_end_token_id", "force_image_size", "norm_mean",
                    "norm_std", "patch_size", "downsample_ratio",
                    "vit_hidden_size", "video_pruning_rate"):
            if key in config:
                vis_cfg_out[key] = config[key]
        # Video sizing lives under vision_config in the official checkpoint (and
        # in vLLM); resolve_video_cfg falls back to top level for older
        # artifacts, then the HF defaults (temporal_patch_size omitted -> 2).
        # Shared with the runtime model build so both agree on T.
        from ..models.nemotron_omni.modeling_nemotron_omni_visual import \
            resolve_video_cfg
        vis_cfg_out["video_temporal_patch_size"] = (resolve_video_cfg(
            config, "video_temporal_patch_size", None) or 2)
        vis_cfg_out["video_target_num_patches"] = resolve_video_cfg(
            config, "video_target_num_patches", 1024)
        vis_cfg_out["video_maintain_aspect_ratio"] = resolve_video_cfg(
            config, "video_maintain_aspect_ratio", True)
    if os.environ.get("USE_TRT_NATIVE_ATTN") == "1":
        vis_cfg_out["use_trt_native_vit_attn"] = True
    cfg_out_path = os.path.join(visual_out_dir, "config.json")
    with open(cfg_out_path, "w") as f:
        json.dump(vis_cfg_out, f, indent=2)
    logger.info("[Visual] Wrote config.json: %s", cfg_out_path)


def _export_alpamayo_visual(model_dir: str, visual_out_dir: str, weights: dict,
                            config: dict, dtype: "torch.dtype",
                            model_config: "ModelConfig") -> None:
    vis_weights, vis_config, vis_model_type = _prepare_alpamayo_visual_params(
        config, weights)
    _export_visual(model_dir,
                   visual_out_dir,
                   vis_weights,
                   vis_config,
                   vis_model_type,
                   dtype,
                   model_config=model_config)
    _save_alpamayo_visual_processor(config, visual_out_dir)


def _export_audio(model_dir: str,
                  audio_out_dir: str,
                  weights: dict,
                  config: dict,
                  model_type: str,
                  dtype: "torch.dtype",
                  model_config: "ModelConfig | None" = None) -> None:
    """Export audio encoder via from-scratch tensorrt_edgellm pipeline."""
    if model_config is not None:
        model_config = _tower_model_config(
            model_config, weights, ("audio_tower.", "thinker.audio_tower.",
                                    "model.audio_tower.", "audio_embed."))
    os.makedirs(audio_out_dir, exist_ok=True)
    output_path = os.path.join(audio_out_dir, "model.onnx")

    logger.info("[Audio] Exporting %s audio encoder to %s", model_type,
                output_path)
    try:
        from ..onnx.export_encoder import export_audio_onnx
        export_audio_onnx(
            model_dir=model_dir,
            output_path=output_path,
            weights=weights,
            config=config,
            model_type=model_type,
            model_config=model_config,
            dtype=dtype,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[Audio] ONNX export failed")
        raise SystemExit(1) from exc
    logger.info("[Audio] Done: %s", output_path)

    # Write config.json for the C++ runtime
    if model_type == "nemotron3_5_asr":
        # Nemotron-3.5-ASR keeps everything the encoder builder AND the RNN-T
        # runtime need (``encoder_config``, ``decoder_hidden_size``,
        # ``blank_token_id``, ``vocab_size``, ``num_decoder_layers``,
        # ``default_prompt_id``, ...) at the config root, so pass it through
        # verbatim. Both the encoder and the RNN-T step share this one file.
        audio_cfg_out = dict(config)
        cfg_out_path = os.path.join(audio_out_dir, "config.json")
        with open(cfg_out_path, "w") as f:
            json.dump(audio_cfg_out, f, indent=2)
        logger.info("[Audio] Wrote config.json: %s", cfg_out_path)
        _copy_asr_tokenizer(model_dir, audio_out_dir)
        return
    if model_type == "gemma4_unified":
        audio_cfg = dict(config.get("audio_config") or {})
        audio_cfg["model_type"] = "gemma4_unified_audio"
        audio_cfg_out = {
            "model_type": "gemma4_unified_audio",
            "audio_config": audio_cfg,
        }
        text_cfg = config.get("text_config") or {}
        if text_cfg:
            audio_cfg_out["text_config"] = text_cfg
        for key in ("audio_token_id", "image_token_id", "boi_token_id",
                    "eoi_token_id", "boa_token_id", "eoa_token_index"):
            if key in config:
                audio_cfg_out[key] = config[key]

        # The source checkpoint keeps raw-waveform framing metadata in the
        # nested feature_extractor section of processor_config.json.  Copy it
        # into the runtime sidecar so the C++ Unified runner does not have to
        # parse an unrelated Hugging Face processor file.
        processor_path = os.path.join(model_dir, "processor_config.json")
        if os.path.exists(processor_path):
            with open(processor_path) as f:
                processor_config = json.load(f)
            feature_extractor = processor_config.get("feature_extractor") or {}
            if feature_extractor:
                audio_cfg_out["feature_extractor"] = feature_extractor
                for key in ("audio_samples_per_token", "sampling_rate",
                            "feature_size", "padding_value"):
                    if key in feature_extractor and key not in audio_cfg:
                        audio_cfg[key] = feature_extractor[key]
    elif model_type in _NEMOTRON_OMNI_MODEL_TYPES:
        # Nemotron-Omni carries ``sound_config`` at the root with its own
        # encoder model_type; keep the full root config alongside so the
        # builder sees everything it needs.
        audio_cfg_out = dict(config)
        sound_model_type = config.get("sound_config", {}).get("model_type")
        if sound_model_type is None:
            raise ValueError(
                "sound_config.model_type not found in config.json")
        audio_cfg_out["model_type"] = sound_model_type
    elif model_type == "gemma4":
        # Gemma4 audio encoder config lives at ``config["audio_config"]``.
        # The C++ audioBuilder requires ``num_mel_bins`` inside audio_config;
        # HF config may omit it (defaults to 128 in Python model code).
        audio_cfg = dict(config.get("audio_config", {}))
        if "num_mel_bins" not in audio_cfg:
            audio_cfg["num_mel_bins"] = 128
        audio_cfg_out = {
            "model_type": "gemma4_audio",
            "audio_config": audio_cfg,
            "builder_config": {
                "max_code_len": 2000,
                "max_time_steps": 6000,
                "min_code_len": 1,
                "min_time_steps": 100,
                "opt_code_len": 300,
            },
        }
        # Propagate audio_token_id from top-level config.
        audio_token_id = config.get("audio_token_id")
        if audio_token_id is None:
            audio_token_id = _find_token_id(model_dir, "<|audio_pad|>")
        if audio_token_id is not None:
            audio_cfg_out["audio_token_id"] = audio_token_id
        # boa/eoa delimiters: gemma4AudioRunner wraps each audio span with
        # them to mirror the HF processor layout (boa + N soft tokens + eoa).
        boa_token_id = config.get("boa_token_id")
        if boa_token_id is None:
            boa_token_id = _find_token_id(model_dir, "<|audio>")
        if boa_token_id is not None:
            audio_cfg_out["boa_token_id"] = boa_token_id
        eoa_token_id = config.get("eoa_token_index",
                                  config.get("eoa_token_id"))
        if eoa_token_id is None:
            eoa_token_id = _find_token_id(model_dir, "<audio|>")
        if eoa_token_id is not None:
            audio_cfg_out["eoa_token_id"] = eoa_token_id
    else:
        # Qwen3-family: read the nested ``audio_config`` and map top-level
        # model_type to the encoder-specific enum the C++ builder expects
        # (``qwen3_asr_thinker``, ``qwen3_omni_audio_encoder``).
        audio_cfg = config.get("thinker_config",
                               {}).get("audio_config",
                                       config.get("audio_config", {}))
        _AUDIO_MODEL_TYPE_MAP = {
            "qwen3_asr": "qwen3_asr_thinker",
            "qwen3_omni": "qwen3_omni_audio_encoder",
            "qwen3_omni_moe": "qwen3_omni_audio_encoder",
            "qwen3_omni_next": "qwen3_omni_next_audio_encoder",
            "qwen3_omni_next_thinker": "qwen3_omni_next_audio_encoder",
        }
        audio_model_type = _AUDIO_MODEL_TYPE_MAP.get(model_type, model_type)
        audio_cfg_out = {
            "model_type": audio_model_type,
            "audio_config": audio_cfg,
        }
        # Propagate multimodal token IDs from ``thinker_config`` to the audio
        # encoder's config.  ``audioRunner.cpp`` reads ``audio_token_id`` from
        # this file to know which placeholder tokens in the prompt to replace
        # with audio-encoder embeddings.  Without it, the runtime falls back
        # to id 0 → no substitution → thinker answers "I can't hear audio".
        thinker_cfg = config.get("thinker_config", {}) or {}
        for key in ("audio_token_id", "audio_start_token_id",
                    "audio_end_token_id"):
            if key in thinker_cfg:
                audio_cfg_out[key] = thinker_cfg[key]
        # ``user_token_id`` for downstream metadata consumers.
        audio_cfg_out.update(_collect_user_token_id(model_dir, config))
        # ``text_config.rope_theta`` is read by
        # ``Qwen3OmniAudioRunner::loadConfig`` for MRope initialisation. For
        # Qwen3-ASR/Omni this lives under ``thinker_config.text_config``.
        text_cfg = (thinker_cfg.get("text_config") or config.get("text_config")
                    or {})
        rope_theta = text_cfg.get("rope_theta")
        if rope_theta is not None:
            audio_cfg_out["text_config"] = {"rope_theta": rope_theta}
    if os.environ.get("USE_TRT_NATIVE_ATTN") == "1":
        audio_cfg_out["use_trt_native_audio_attn"] = True
    cfg_out_path = os.path.join(audio_out_dir, "config.json")
    with open(cfg_out_path, "w") as f:
        json.dump(audio_cfg_out, f, indent=2)
    logger.info("[Audio] Wrote config.json: %s", cfg_out_path)


def _copy_asr_tokenizer(model_dir: str, out_dir: str) -> None:
    """Copy the RNN-T tokenizer and prompt metadata into the export dir.

    ``NemotronAsrRuntime`` detokenizes emitted RNN-T tokens with
    ``tokenizer.json`` (``tokenizer_config.json`` is optional — special-token
    config). Copying them here keeps the exported engine dir self-contained.
    """
    import shutil
    required = ("tokenizer.json", "processor_config.json")
    missing = [
        name for name in required
        if not os.path.isfile(os.path.join(model_dir, name))
    ]
    if missing:
        raise FileNotFoundError(
            "Nemotron-3.5-ASR checkpoint is missing required runtime "
            f"artifacts: {', '.join(missing)}")
    for name in ("tokenizer.json", "tokenizer_config.json",
                 "processor_config.json"):
        src = os.path.join(model_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, name))
            logger.info("[Audio] Copied %s", name)


# ---------------------------------------------------------------------------
# RNN-T decoder-step export (Nemotron-3.5-ASR)
# ---------------------------------------------------------------------------


def _export_rnnt_decoder(model_dir: str, out_dir: str, weights: dict,
                         config: dict, dtype: "torch.dtype") -> None:
    """Export the fused RNN-T step (LSTM prediction network + joint) to ONNX.

    One decode step: ``(decoder_input_ids, hidden_state, cell_state,
    encoder_frame) -> (logits, present_hidden_state, present_cell_state)``.
    All-static shapes (the greedy loop lives in the C++ runtime).
    """
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "model.onnx")
    logger.info("[RNN-T] Exporting decoder step to %s", output_path)
    try:
        from ..models.nemotron3_5_asr import build_nemotron3_5_asr_decoder
        from ..onnx.export_encoder import _run_dynamo_export
        step = build_nemotron3_5_asr_decoder(config, weights, dtype=dtype)
        step = step.to("cpu").eval()
        args_, input_names, output_names, dynamic_shapes = (
            step.get_onnx_export_args(config, "cpu"))
        _run_dynamo_export(step, args_, output_path, input_names, output_names,
                           dynamic_shapes)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[RNN-T] ONNX export failed")
        raise SystemExit(1) from exc
    logger.info("[RNN-T] Done: %s", output_path)

    # Build-type marker sidecar: the C++ audioBuilder auto-detects the RNN-T
    # step build from the ``rnnt_decoder_config`` key (the engine itself is
    # fully static-shape). The runtime reads the full config from the encoder
    # dir; the fields here are informational.
    step_cfg = {
        "model_type": "nemotron3_5_asr",
        "rnnt_decoder_config": {
            "blank_token_id": config.get("blank_token_id"),
            "vocab_size": config.get("vocab_size"),
            "decoder_hidden_size": config.get("decoder_hidden_size"),
            "num_decoder_layers": config.get("num_decoder_layers", 2),
            "max_symbols_per_step": config.get("max_symbols_per_step", 10),
        },
    }
    cfg_out_path = os.path.join(out_dir, "config.json")
    with open(cfg_out_path, "w") as f:
        json.dump(step_cfg, f, indent=2)
    logger.info("[RNN-T] Wrote config.json: %s", cfg_out_path)


# ---------------------------------------------------------------------------
# Code2Wav export
# ---------------------------------------------------------------------------


def _export_code2wav(model_dir: str, c2w_out_dir: str, weights: dict,
                     config: dict, model_type: str,
                     dtype: "torch.dtype") -> None:
    """Export Code2Wav vocoder via the standalone tensorrt_edgellm implementation.

    The vocoder converts discrete RVQ codec tokens
    ``[batch, num_quantizers, code_length]`` into continuous audio
    waveforms ``[batch, 1, code_length * total_upsample]``.

    Qwen3-Omni stores Code2Wav weights in the shared checkpoint under the
    ``code2wav.`` prefix. Qwen3-TTS stores the vocoder in the
    ``speech_tokenizer/`` subdirectory.
    """
    os.makedirs(c2w_out_dir, exist_ok=True)
    output_path = os.path.join(c2w_out_dir, "model.onnx")

    if model_type == "qwen3_tts":
        logger.info("[Code2Wav] Exporting Qwen3-TTS speech tokenizer")
        try:
            from ..models.qwen3_tts import export_qwen3_tts_code2wav
            export_qwen3_tts_code2wav(model_dir, c2w_out_dir, dtype)
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            logger.exception("[Code2Wav] Qwen3-TTS export failed")
            raise SystemExit(1) from exc
        logger.info("[Code2Wav] Done: %s", output_path)
        if config.get("tts_model_type") == "base":
            # Base checkpoints clone voices from reference audio: export the
            # reference encoders (ECAPA x-vector + Mimi codec encoder) too.
            logger.info(
                "[CloneEncoders] Exporting voice-clone reference encoders")
            try:
                from ..models.qwen3_tts import export_qwen3_tts_clone_encoders
                clone_out_dir = os.path.join(os.path.dirname(c2w_out_dir),
                                             "clone_encoders")
                export_qwen3_tts_clone_encoders(model_dir, clone_out_dir)
            except (OSError, ValueError, RuntimeError, ImportError) as exc:
                logger.exception("[CloneEncoders] export failed")
                raise SystemExit(1) from exc
        return

    if model_type == "qwen3_omni_next":
        # Qwen3-Next Omni Code2Wav config + weights live in a separate directory
        # (release name ``codec_decode_online/`` shipped alongside the HF
        # checkpoint), containing ``config.yaml`` + ``model_weights.pt``.
        # Its architecture (SplitResidualVectorQuantizer + Llama-style
        # WindowLimitedTransformer) is incompatible with Qwen3-Omni's vocoder.
        c2w_dir = os.environ.get(
            "QWEN3_OMNI_NEXT_CODE2WAV_DIR") or os.path.join(
                model_dir, "codec_decode_online")
        if not (os.path.isfile(os.path.join(c2w_dir, "config.yaml"))
                and os.path.isfile(os.path.join(c2w_dir, "model_weights.pt"))):
            logger.error(
                "[Code2Wav] Qwen3-Next Omni vocoder expected config.yaml + "
                "model_weights.pt under %r (override with the "
                "QWEN3_OMNI_NEXT_CODE2WAV_DIR env var). Skip with "
                "--skip-code2wav.", c2w_dir)
            sys.exit(1)
        logger.info("[Code2Wav] Building Qwen3-Next Omni model from %s",
                    c2w_dir)
        try:
            from ..models.qwen3_omni_next import (
                build_qwen3_omni_next_code2wav,
                export_qwen3_omni_next_code2wav_onnx)
            model = build_qwen3_omni_next_code2wav(c2w_dir,
                                                   dtype=dtype).to("cuda")
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            logger.exception("[Code2Wav] Failed to build model")
            raise SystemExit(1) from exc

        logger.info("[Code2Wav] Exporting ONNX to %s", output_path)
        try:
            export_qwen3_omni_next_code2wav_onnx(model, output_path)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.exception("[Code2Wav] ONNX export failed")
            raise SystemExit(1) from exc

        # Write a config.json the C++ runtime can consume.  ``code2wav_config``
        # mirrors the dataclass defaults in ``Code2WavConfig`` (n_q=16,
        # codebook_size=2048, decoder_dim=1536) and exposes the two upsample
        # lists that the runtime multiplies to compute samples-per-code
        # (product([2, 2]) * product([8, 5, 4, 3]) = 1920 samples / code).
        c2w_cfg_out = {
            "num_quantizers": 16,
            "codebook_size": 2048,
            "hidden_size": 1024,
            "decoder_dim": 1536,
            "upsample_rates": [2, 2],
            "upsampling_ratios": [8, 5, 4, 3],
            "sample_rate": 24000,
        }
        cfg_out_path = os.path.join(c2w_out_dir, "config.json")
        with open(cfg_out_path, "w") as f:
            json.dump(
                {
                    "model_type": "qwen3_omni_next_code2wav",
                    "code2wav_config": c2w_cfg_out,
                    "builder_config": {
                        "max_code_len": 2000,
                        "min_code_len": 1,
                        "opt_code_len": 300,
                    },
                },
                f,
                indent=2)
        logger.info("[Code2Wav] Wrote config.json: %s", cfg_out_path)
        logger.info("[Code2Wav] Done: %s", output_path)
        return

    c2w_cfg = config.get("code2wav_config")
    if not c2w_cfg:
        logger.error(
            "code2wav_config not found in config.json — cannot export Code2Wav"
        )
        sys.exit(1)

    logger.info("[Code2Wav] Building model and loading weights")
    try:
        from ..models.qwen3_omni import build_code2wav, export_code2wav_onnx
        model = build_code2wav(c2w_cfg, weights, dtype)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.exception("[Code2Wav] Failed to build model")
        raise SystemExit(1) from exc

    logger.info("[Code2Wav] Exporting ONNX to %s", output_path)
    try:
        export_code2wav_onnx(model, output_path, c2w_cfg)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[Code2Wav] ONNX export failed")
        raise SystemExit(1) from exc

    # Write a config.json that the C++ runtime / engine builder can consume.
    # Match the layout produced by tensorrt_edgellm.export_code2wav_config:
    # top-level model_type is "qwen3_omni_code2wav" and the sub-config
    # carries the same model_type for parser compatibility.
    c2w_cfg_out = dict(c2w_cfg)
    c2w_cfg_out["model_type"] = "qwen3_omni_code2wav"
    cfg_out_path = os.path.join(c2w_out_dir, "config.json")
    with open(cfg_out_path, "w") as f:
        json.dump(
            {
                "model_type": "qwen3_omni_code2wav",
                "code2wav_config": c2w_cfg_out,
            },
            f,
            indent=2)
    logger.info("[Code2Wav] Wrote config.json: %s", cfg_out_path)
    logger.info("[Code2Wav] Done: %s", output_path)


# ---------------------------------------------------------------------------
# TTS Talker export
# ---------------------------------------------------------------------------


def _projection_mlp_specs(ckpt_prefix: str) -> list:
    """Return ``[(save_name, ckpt_key), ...]`` for the shared 2-layer MLP
    projection layout (``linear_fc1`` + ``linear_fc2``, weight + bias).
    """
    return [(f"{fc}.{a}", f"{ckpt_prefix}.{fc}.{a}")
            for fc in ("linear_fc1", "linear_fc2") for a in ("weight", "bias")]


def _extract_tts_weights(model_dir: str, out_dir: str) -> None:
    """Extract Qwen3-TTS talker sidecars (``text_embedding`` + ``text_projection``)."""
    _extract_sidecars(
        model_dir,
        out_dir,
        [
            ("text_embedding.safetensors", [
                ("text_embedding", "talker.model.text_embedding.weight")
            ], True),
            ("text_projection.safetensors",
             _projection_mlp_specs("talker.text_projection"), True),
        ],
        strict=True,
    )


def _extract_omni_talker_sidecars(model_dir: str, out_dir: str) -> None:
    """Extract Qwen3-Omni Talker sidecars.

    Qwen3-Omni Talker consumes the thinker's ``hidden_states`` directly, so
    ships three sidecars: ``embedding`` (codec token embedding),
    ``hidden_projection`` (thinker hidden space → talker space; Omni-only),
    ``text_projection`` (shared with Qwen3-TTS).
    """
    _extract_sidecars(
        model_dir,
        out_dir,
        [
            ("embedding.safetensors", [
                ("embedding", "talker.model.codec_embedding.weight")
            ], True),
            ("hidden_projection.safetensors",
             _projection_mlp_specs("talker.hidden_projection"), True),
            ("text_projection.safetensors",
             _projection_mlp_specs("talker.text_projection"), True),
        ],
        strict=True,
    )


def _make_talker_sub_config(model_dir: str,
                            sub_path=None,
                            *,
                            sub_cfg: "Optional[dict]" = None,
                            key_prefix: str = "",
                            key_remap=None) -> "ModelConfig":
    """Build a :class:`ModelConfig` from a nested sub-config of ``config.json``.

    Multi-stage Talker/CodePredictor exports share this helper: the dense
    decoder's architecture config lives under ``talker_config.*`` (or a caller
    passes ``sub_cfg`` directly for families whose config is already extracted).
    Writes the sub-dict into a temp ``config.json`` and symlinks the root's
    safetensors so quant detection still works.

    When ``key_prefix`` is provided the temp dir's ``hf_quant_config.json``
    ``exclude_modules`` list is rewritten (prefix stripped, optional key_remap
    applied) so exclusion globs match the sub-LLM's short module paths.  If the
    entire sub-LLM is excluded (glob collapses to ``*``) the sidecar is dropped
    so ``_parse_quant`` returns the default FP16 config.

    Args:
        model_dir:  Directory containing the checkpoint's root ``config.json``.
        sub_path:   Sequence of keys to walk into the root config, e.g.
                    ``["talker_config", "text_config"]``.  Ignored when
                    ``sub_cfg`` is passed.
        sub_cfg:    Fully-formed sub-config dict (bypasses the ``sub_path`` walk).
        key_prefix: Checkpoint-side prefix (e.g. ``"talker."``) to strip from
                    quant exclusion patterns.
        key_remap:  Optional ``str -> Optional[str]`` fn applied to each
                    exclusion pattern after prefix stripping.
    """
    import tempfile

    from ..model import load_model_config

    root_cfg = _load_config(model_dir)
    root_quant = root_cfg.get("quantization_config")
    if sub_cfg is None:
        cfg = root_cfg
        for key in sub_path or ():
            if not isinstance(cfg, dict) or key not in cfg:
                logger.error("sub-config path %s not found in %s/config.json",
                             ".".join(sub_path), model_dir)
                sys.exit(1)
            cfg = cfg[key]
    else:
        cfg = sub_cfg
    # Quant metadata lives at the root config only; the sub-model inherits it.
    if root_quant is not None and "quantization_config" not in cfg:
        cfg = {**cfg, "quantization_config": root_quant}

    with tempfile.TemporaryDirectory() as tmp_dir:
        for fname in os.listdir(model_dir):
            if (fname.endswith(".safetensors")
                    or fname.endswith(".safetensors.index.json")):
                if key_prefix and fname.endswith(".safetensors.index.json"):
                    continue  # replaced by the filtered index below
                src = os.path.join(model_dir, fname)
                dst = os.path.join(tmp_dir, fname)
                if not os.path.exists(dst):
                    os.symlink(src, dst)
        if key_prefix:
            # Quant detection (``_detect_modelopt_unquantized_linears`` /
            # ``_effective_excluded_modules``) scans checkpoint tensor names.
            # On a multi-component root the OTHER components' unquantized
            # modules would collide with this sub-LLM's short module names
            # after normalization (e.g. the Talker body's bf16
            # ``model.layers.3.self_attn.q_proj`` masks the CodePredictor's
            # quantized layer 3) — stage an index restricted to this
            # component, with the prefix stripped.
            _stage_component_weight_index(model_dir, tmp_dir, key_prefix)
        fully_excluded = _maybe_stage_hf_quant_config(model_dir, tmp_dir,
                                                      key_prefix, key_remap)
        if fully_excluded and "quantization_config" in cfg:
            # The whole sub-LLM sits in the FP16 keep-set: the root-level
            # quantization_config it inherited would rebuild every Linear
            # as a quantized class over unquantized weights.
            cfg = {k: v for k, v in cfg.items() if k != "quantization_config"}
        with open(os.path.join(tmp_dir, "config.json"), "w") as f:
            json.dump(cfg, f)
        return load_model_config(tmp_dir)


def _stage_component_weight_index(model_dir: str, tmp_dir: str,
                                  key_prefix: str) -> None:
    """Write a ``model.safetensors.index.json`` covering only *key_prefix*.

    Keys are prefix-stripped so they match the sub-LLM's module paths; shard
    filenames stay valid via the safetensors symlinks staged alongside. Also
    handles single-file checkpoints (synthesizes an index over
    ``model.safetensors``).
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")
    weight_map = {}
    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f).get("weight_map", {})
    elif os.path.isfile(single_path):
        from safetensors import safe_open
        with safe_open(single_path, framework="pt") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}
    filtered = {
        k[len(key_prefix):]: shard
        for k, shard in weight_map.items() if k.startswith(key_prefix)
    }
    if not filtered:
        return
    with open(os.path.join(tmp_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": filtered}, f)


def _sub_llm_has_quantized_weights(model_dir: str, key_prefix: str) -> bool:
    """True if *model_dir* ships any quantized tensor under *key_prefix*.

    NVFP4/FP8 linears carry a ``weight_scale`` sidecar; AWQ/GPTQ instead pack
    the weight itself as ``qweight`` and carry no ``weight_scale``, so both
    markers have to be checked or those sub-LLMs read as fully FP16. Reads the
    safetensors index / shard headers only.
    """
    import glob
    import struct
    markers = (".weight_scale", ".qweight")

    def _hit(keys) -> bool:
        return any(
            k.startswith(key_prefix) and k.endswith(markers) for k in keys)

    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            return _hit(json.load(f).get("weight_map", {}))
    for sf in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(sf, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        if _hit(hdr):
            return True
    return False


def _maybe_stage_hf_quant_config(model_dir: str, tmp_dir: str, key_prefix: str,
                                 key_remap) -> bool:
    """Rewrite ``hf_quant_config.json``'s ``exclude_modules`` for a sub-LLM.

    Drops patterns that belong to other sub-LLMs (don't start with
    *key_prefix*), strips the prefix from surviving patterns, applies
    *key_remap*, and skips the file entirely when the whole sub-LLM is
    excluded (glob becomes ``*``).
    """
    hf_qc_src = os.path.join(model_dir, "hf_quant_config.json")
    if not os.path.isfile(hf_qc_src):
        # Modelopt roots keep the quant metadata in config.json's
        # ``quantization_config`` instead (exact module names, not a ``*``
        # glob). A sub-LLM is fully FP16 iff it ships no quantized tensor, so
        # detect that directly: no ``{key_prefix}*.weight_scale`` in the
        # checkpoint means the inherited quantization_config must be dropped.
        if key_prefix and not _sub_llm_has_quantized_weights(
                model_dir, key_prefix):
            return True
        return False
    if not key_prefix:
        # No sub-LLM namespace: the exclusion patterns already match the
        # checkpoint keys — stage the sidecar as-is (modelopt-quantized
        # consolidated roots keep their quant metadata there).
        dst = os.path.join(tmp_dir, "hf_quant_config.json")
        if not os.path.exists(dst):
            os.symlink(hf_qc_src, dst)
        return False
    with open(hf_qc_src) as f:
        hf_qc = json.load(f)
    stripped_prefix = key_prefix.rstrip(".")
    excl = hf_qc.get("quantization", {}).get("exclude_modules", [])
    new_excl = []
    for pat in excl:
        if pat.startswith(key_prefix):
            pat = pat[len(key_prefix):]
        elif pat.startswith(stripped_prefix):
            pat = pat[len(stripped_prefix):]
        elif not (pat.startswith("*") or pat == ""):
            # A leading ``*`` makes the glob sub-LLM agnostic (e.g. ``*lm_head*``);
            # anything else that is not our prefix names another sub-LLM. Testing
            # for ``*`` anywhere would keep ``talker.…linear_attn*`` in the
            # thinker's list, where prefix normalisation collapses it onto the
            # thinker's own modules.
            continue
        if key_remap is not None and pat:
            remapped = key_remap(pat)
            if remapped is not None:
                pat = remapped
        new_excl.append(pat)
    if "*" in new_excl:
        return True  # entire sub-LLM unquantized → skip sidecar entirely
    hf_qc.setdefault("quantization", {})["exclude_modules"] = new_excl
    with open(os.path.join(tmp_dir, "hf_quant_config.json"), "w") as f:
        json.dump(hf_qc, f)
    return False


def _patch_tts_config(model_dir: str, out_dir: str) -> None:
    """Patch the exported config.json with TTS-specific fields.

    Reads the input config and injects codec token IDs, TTS token IDs,
    thinker_hidden_size, and speaker_id mapping into the already-written
    config.json in *out_dir*.

    Accepts two input layouts:
      * HF root config with nested ``talker_config`` (fields under sub-dict).
      * Standalone Talker config from an older split-checkpoint export
        (fields at top level).
    """
    root_config = _load_config(model_dir)
    # ``or {}`` guards against explicit ``"talker_config": null`` in a
    # standalone Talker config (where ``.get`` returns None, not {}).
    talker_cfg = root_config.get("talker_config") or {}

    def pick(key, *fallback_keys):
        """Lookup *key* with fallback: talker_cfg -> root_config -> fallbacks."""
        if key in talker_cfg:
            return talker_cfg[key]
        if key in root_config:
            return root_config[key]
        for fb in fallback_keys:
            if fb in talker_cfg:
                return talker_cfg[fb]
            if fb in root_config:
                return root_config[fb]
        return None

    cfg_path = os.path.join(out_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    # TTS token IDs and Codec token IDs and runtime knobs:
    # each lives at top level of the standalone Talker config OR under
    # ``talker_config`` of the HF root config.
    for key in ("tts_pad_token_id", "tts_bos_token_id", "tts_eos_token_id",
                "codec_nothink_id", "codec_think_bos_id", "codec_think_eos_id",
                "codec_pad_id", "codec_bos_id", "codec_eos_token_id",
                "codec_think_id", "accept_hidden_layer", "num_code_groups"):
        v = pick(key)
        if v is not None:
            cfg[key] = v

    # tts_model_type (only present in root config, not talker_config).
    if "tts_model_type" in root_config:
        cfg["tts_model_type"] = root_config["tts_model_type"]

    # thinker_hidden_size (Qwen3-Omni) or text_hidden_size (Qwen3-TTS naming).
    thinker_hs = pick("thinker_hidden_size", "text_hidden_size")
    if thinker_hs is not None:
        cfg["thinker_hidden_size"] = thinker_hs

    # ``text_vocab_size`` is the thinker's text vocab; Qwen3-TTS exposes it
    # on talker_config; Qwen3-Omni nests it under thinker_config.text_config.
    tv = pick("text_vocab_size")
    if tv is None:
        thinker_cfg = root_config.get("thinker_config", {}) or {}
        thinker_text = thinker_cfg.get("text_config", {}) or {}
        tv = thinker_text.get("vocab_size") or thinker_cfg.get("vocab_size")
    if tv is not None:
        cfg["text_vocab_size"] = tv

    # The Talker is a text-only decoder with no deepstack visual inputs.
    # Override the (potentially inherited) value from ModelConfig which may
    # mistakenly set 3 due to substring-matching on "qwen3_omni" in the
    # talker sub-config's model_type (``qwen3_omni_talker_text``).
    cfg["num_deepstack_features"] = 0

    # Speaker ID mapping. Qwen3-TTS stores the name→id dict under ``spk_id``,
    # Qwen3-Omni under ``speaker_id`` (naming convention drift). Accept
    # either; missing default_speaker_id causes TTS runtime to pick token 0
    # as speaker → Talker generates garbage / fails to emit codec EOS.
    spk_map = pick("speaker_id", "spk_id")
    if isinstance(spk_map, dict) and spk_map:
        cfg["speaker_id"] = spk_map
        if "default_speaker_id" not in cfg:
            cfg["default_speaker_id"] = next(iter(spk_map.values()))
    # Also propagate an explicit ``default_speaker_id`` if the input
    # already had one (e.g. from an older split-checkpoint export).
    dsi = pick("default_speaker_id")
    if dsi is not None and "default_speaker_id" not in cfg:
        cfg["default_speaker_id"] = dsi

    # CustomVoice language conditioning. Qwen3-TTS stores the name→codec-id
    # dict under ``talker_config.codec_language_id``; some checkpoints keep it
    # at root level as ``talker_language_id`` (pick's root fallback covers both).
    # Absent maps are not written (no ``null`` keys) — the runtime then keeps
    # the no-language prefill.
    lang_map = pick("codec_language_id", "talker_language_id")
    if isinstance(lang_map, dict) and lang_map:
        cfg["codec_language_id"] = lang_map
    # Dialect speaker map: values are false (non-dialect) or a dialect name
    # string; forwarded verbatim so the runtime can apply the PyTorch
    # dialect-override rule for speakers like eric/dylan.
    dialect_map = pick("spk_is_dialect")
    if isinstance(dialect_map, dict) and dialect_map:
        cfg["spk_is_dialect"] = dialect_map

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("[TTS] Patched config.json with TTS/codec fields")


def _talker_key_remap(key: str) -> "Optional[str]":
    """Rename talker checkpoint keys so they match :class:`CausalLM`'s
    expected structure.

    Both Qwen3-TTS and Qwen3-Omni need ``codec_embedding`` → ``embed_tokens``.
    Qwen3-Omni additionally has ``codec_head`` (output head for codec tokens)
    which must be renamed to ``lm_head``.  Qwen3-TTS checkpoints don't
    contain ``codec_head`` so the second branch is a no-op for them —
    the same remap is safe to use for both model families.
    """
    if "codec_embedding" in key:
        key = key.replace("codec_embedding", "embed_tokens")
    if "codec_head" in key:
        key = key.replace("codec_head", "lm_head")
    return key


# ---------------------------------------------------------------------------
# Qwen3-Next Omni (qwen3_omni_next) Talker + CodePredictor helpers
# ---------------------------------------------------------------------------
#
# Qwen3-Next Omni's Talker and CodePredictor are Qwen3.5-gated decoders (Talker is
# hybrid GDN + gated full-attention; CP is a 5-layer dense gated decoder), so
# the upstream qwen3_tts ``TalkerCausalLM`` / ``CodePredictorCausalLM`` cannot
# be reused — they target plain Qwen3 attention.  The helpers below stage a
# sub-LLM tmp dir from the nested ``talker_config.text_config`` /
# ``talker_config.code_predictor_config`` so :class:`ModelConfig` can parse
# them as standalone checkpoints, then build the Qwen3.5-specific model class
# from ``models.qwen3_omni_next`` and load weights through the standard
# checkpoint loader with a key_prefix + key_remap.


def _export_sub_llm(
    model_dir: str,
    out_dir: str,
    *,
    model_class,
    sub_path=None,
    sub_config: "Optional[dict]" = None,
    key_prefix: str = "",
    key_remap=None,
    model_type_override: "Optional[str]" = None,
) -> None:
    """Build a sub-LLM (Talker / CodePredictor), load its weights, and export.

    Either ``sub_path`` (walk nested root config) or ``sub_config`` (already
    extracted dict) selects the sub-LLM config; when both are ``None`` the
    root ``config.json`` is used directly.
    """
    os.makedirs(out_dir, exist_ok=True)

    from ..checkpoint.loader import load_weights
    from ..model import load_model_config
    from ..onnx.export import export_onnx

    tag = os.path.basename(out_dir) or "SubLLM"
    logger.info("[%s] Loading checkpoint from %s", tag, model_dir)
    try:
        if sub_path is None and sub_config is None:
            config = load_model_config(model_dir)
        else:
            config = _make_talker_sub_config(model_dir,
                                             sub_path,
                                             sub_cfg=sub_config,
                                             key_prefix=key_prefix,
                                             key_remap=key_remap)

        if model_type_override:
            config.model_type = model_type_override

        model = model_class(config)
        load_weights(model,
                     model_dir,
                     device="cpu",
                     key_prefix=key_prefix,
                     key_remap=key_remap)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.exception("[%s] Failed to load checkpoint", tag)
        raise SystemExit(1) from exc

    output_path = os.path.join(out_dir, "model.onnx")
    logger.info("[%s] Exporting ONNX to %s", tag, output_path)
    try:
        export_onnx(model, output_path, model_dir=model_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[%s] ONNX export failed", tag)
        raise SystemExit(1) from exc
    logger.info("[%s] Done: %s", tag, output_path)


def _extract_sidecars(model_dir: str,
                      out_dir: str,
                      specs,
                      *,
                      strict: bool = False) -> None:
    """Dump selected checkpoint tensors into sidecar safetensors files.

    *specs*: iterable of ``(filename, [(save_name, ckpt_key), ...], fp16_cast)``.
    ``fp16_cast=True`` casts bfloat16->float16 for C++ runtime compatibility.

    With ``strict=True`` a missing checkpoint key is a fatal error (matches
    the historic Qwen3-Omni / Qwen3-TTS extractors' behaviour); with the
    default ``strict=False`` a missing key is logged as a warning and the
    file is skipped if no tensors survive.
    """
    from tensorrt_edgellm._safetensors_io import save_file

    weights = _load_all_weights(model_dir)
    for filename, keys, fp16 in specs:
        tensors: dict = {}
        for save_name, ckpt_key in keys:
            t = weights.get(ckpt_key)
            if t is None:
                if strict:
                    logger.error("Key %r not found in checkpoint", ckpt_key)
                    sys.exit(1)
                logger.warning("[Sidecar] %r missing for %s", ckpt_key,
                               filename)
                continue
            tensors[save_name] = t.cpu()
        if not tensors:
            continue
        out = _to_fp16(tensors) if fp16 else tensors
        save_file(out, os.path.join(out_dir, filename))
        first_shape = list(next(iter(tensors.values())).shape)
        logger.info("[Sidecar] Wrote %s (%d tensors, first shape %s)",
                    filename, len(tensors), first_shape)


def _write_downcast_fp16_sidecar(model_dir: str,
                                 out_dir: str,
                                 filename: str,
                                 keys: list,
                                 *,
                                 strict: bool = False) -> None:
    """Extract *keys* from the checkpoint and write them as fp16 safetensors.

    Unlike :func:`_extract_sidecars`, downcasts ``float32`` (as well as
    ``bfloat16``) source tensors so the C++ runtime's ``__half*`` reader
    always sees fp16. Reserved for sidecars whose HF source is fp32 — the
    shared ``_to_fp16`` helper stays bfloat16-only.

    With ``strict=True`` a missing checkpoint key is a fatal error
    (matches :func:`_extract_sidecars`).
    """
    import torch

    from tensorrt_edgellm._safetensors_io import save_file
    weights = _load_all_weights(model_dir)
    out: dict = {}
    for save_name, ckpt_key in keys:
        t = weights.get(ckpt_key)
        if t is None:
            if strict:
                logger.error("Key %r not found in checkpoint", ckpt_key)
                sys.exit(1)
            logger.warning("[Sidecar] %r missing for %s", ckpt_key, filename)
            continue
        if t.dtype == torch.float8_e4m3fn:
            # CP-FP8 quantization also covers this projection; the C++
            # runtime reads the sidecar as plain __half, so dequantize
            # with the per-channel weight_scale instead of shipping raw
            # FP8 bytes (and silently dropping the scale).
            scale = weights.get(f"{ckpt_key}_scale")
            if scale is None:
                logger.error("FP8 tensor %r has no %s_scale in checkpoint",
                             ckpt_key, ckpt_key)
                sys.exit(1)
            t = t.to(torch.float32) * scale.reshape(-1, *([1] * (t.dim() - 1)))
        if t.dtype in (torch.bfloat16, torch.float32):
            t = t.to(torch.float16)
        out[save_name] = t.cpu()
    if not out:
        return
    save_file(out, os.path.join(out_dir, filename))
    first_shape = list(next(iter(out.values())).shape)
    logger.info("[Sidecar] Wrote %s (%d tensors, first shape %s)", filename,
                len(out), first_shape)


def _patch_exported_config(
        out_dir: str,
        root_config: dict,
        *,
        copy_from_root=(),
        copy_from_talker=(),
        extra: "Optional[dict]" = None,
) -> None:
    """Merge extra fields into an already-written ``config.json``.

    *copy_from_root* / *copy_from_talker* items are either a plain key string
    (copy by same name) or a ``(src, dst)`` tuple (rename).  Also normalises
    ``speaker_id`` / ``spk_id`` and sets ``default_speaker_id`` from the first
    speaker, which is common to every TTS-style export.
    """
    cfg_path = os.path.join(out_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    def _apply(src_dict, spec):
        for item in spec:
            src, dst = item if isinstance(item, tuple) else (item, item)
            if src in src_dict:
                cfg[dst] = src_dict[src]

    _apply(root_config, copy_from_root)
    talker = root_config.get("talker_config", {}) or {}
    _apply(talker, copy_from_talker)

    spk = talker.get("speaker_id") or talker.get("spk_id")
    if isinstance(spk, dict) and spk:
        cfg["speaker_id"] = spk
        cfg.setdefault("default_speaker_id", next(iter(spk.values())))

    if extra:
        cfg.update(extra)

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("[Config] Patched %s", cfg_path)


def _export_omni_next_talker(model_dir: str, out_dir: str) -> None:
    """Export Qwen3-Next Omni Talker (24-layer hybrid GDN + gated-attention
    decoder that additionally emits hidden_states for the CP residual) +
    text_embedding / hidden_projection / speaker_codec_embeddings sidecars.
    """
    from ..models.qwen3_omni_next import (Qwen3OmniNextMoeTalkerCausalLM,
                                          Qwen3OmniNextTalkerCausalLM)

    root = _load_config(model_dir)
    t_cfg = dict((root.get("talker_config") or {}).get("text_config") or {})
    if not t_cfg.get("hidden_size"):
        logger.error("talker_config.text_config not found in config.json")
        sys.exit(1)

    # The Talker ships in both dense and sparse-MoE variants under the same
    # talker_config. Select the MoE backbone when the checkpoint declares
    # routed experts (Qwen3-Omni Next); otherwise the dense one.
    talker_is_moe = int(t_cfg.get("num_experts", 0) or 0) > 0
    if talker_is_moe:
        talker_model_class = Qwen3OmniNextMoeTalkerCausalLM
        talker_model_type = "qwen3_omni_next_talker_text"
    else:
        talker_model_class = Qwen3OmniNextTalkerCausalLM
        talker_model_type = "qwen3_omni_next_talker"

    # HF auto-generates layer_types when None: every 4th layer is
    # full_attention, the rest are linear_attention (3:1 GDN/full interleave).
    if t_cfg.get("layer_types") is None:
        n = t_cfg["num_hidden_layers"]
        t_cfg["layer_types"] = [
            "linear_attention" if bool((i + 1) % 4) else "full_attention"
            for i in range(n)
        ]

    _export_sub_llm(
        model_dir,
        out_dir,
        model_class=talker_model_class,
        sub_config=t_cfg,
        key_prefix="talker.",
        key_remap=_talker_key_remap,
        model_type_override=talker_model_type,
    )
    _extract_sidecars(
        model_dir,
        out_dir,
        [
            ("text_embedding.safetensors", [
                ("text_embedding", "talker.model.embed_tokens.weight")
            ], True),
            # Talker has TWO embedding tables: ``model.embed_tokens`` (text, 248320 vocab)
            # and ``model.codec_embedding`` (codec, 5120 vocab). The HF ``_get_talker_*_parts``
            # builders use the codec table for codec_bos/eos/think/pad token lookups, while
            # the LLM model class only has a single ``embed_tokens`` slot (the key remap
            # collapses codec_embedding -> embed_tokens for the dense backbone). So the codec
            # table is dumped as a separate sidecar for the C++ runtime to load — without
            # this file the runtime indexes codec tokens (0..5119) into the text embed and
            # the Talker produces saturated noise instead of speech.
            ("codec_embedding.safetensors", [
                ("codec_embedding", "talker.model.codec_embedding.weight")
            ], True),
            ("speaker_codec_embeddings.safetensors", [
                ("speaker_codec_embeddings", "talker.speaker_codec_embeddings")
            ], False),  # int64 LUT, don't cast
        ],
        strict=True)
    # hidden_projection ships in fp32 in HF; C++ TalkerRunner reads it as
    # __half. Extract + downcast manually so the shared ``_to_fp16``
    # helper keeps its bfloat16-only contract.
    _write_downcast_fp16_sidecar(
        model_dir,
        out_dir,
        "hidden_projection.safetensors", [
            ("weight", "talker.hidden_projection.weight"),
            ("bias", "talker.hidden_projection.bias"),
        ],
        strict=True)
    _patch_exported_config(
        out_dir,
        root,
        copy_from_root=("tts_pad_token_id", "tts_bos_token_id",
                        "tts_eos_token_id", "max_thinker_to_talker_mm_tokens",
                        "talker_language_id",
                        "talker_assistant_prompt_id_mapping"),
        copy_from_talker=(
            "codec_nothink_id",
            "codec_think_bos_id",
            "codec_think_eos_id",
            "codec_pad_id",
            "codec_bos_id",
            "codec_eos_token_id",
            "codec_think_id",
            "accept_hidden_layer",
            "num_code_groups",
            "thinker_hidden_size",
            # Per-speaker system prompt rows (~13 tokens each):
            # HF _get_talker_system_parts inserts them between the
            # system role header and codec_bos; omitting them drops
            # the speaker instruction from every Talker prompt.
            "speaker_system_prompt_id"),
    )
    # Friendly speaker aliases ("Ryan" → "m36") consumed by the TTS runtime.
    voice_map_src = os.path.join(model_dir, "voice_map.json")
    if os.path.exists(voice_map_src):
        import shutil
        shutil.copy2(voice_map_src, os.path.join(out_dir, "voice_map.json"))
        logger.info("[Sidecar] Copied voice_map.json")


def _export_omni_next_code_predictor(model_dir: str, out_dir: str) -> None:
    """Export Qwen3-Next Omni CodePredictor (5-layer gated-attention dense
    decoder, head_dim 256, partial_rotary 0.25, MRope interleaved).

    ``model_dir`` is either the full Omni HF root
    (``talker_config.code_predictor_config``, keys
    ``talker.code_predictor.*``) or a Talker-root checkpoint produced by the
    CP-FP8 quantization pass (``code_predictor_config`` at root, keys
    ``code_predictor.*``). Mirrors the Qwen3-Omni CP export's dual-layout
    handling; the FP8 QDQ flows in via ``hf_quant_config.json``
    (``_export_sub_llm`` re-namespaces it with ``key_prefix``).
    """
    from ..models.qwen3_omni_next import Qwen3OmniNextCodePredictorCausalLM

    root = _load_config(model_dir)
    talker = root.get("talker_config", {}) or {}
    cp_cfg = dict(talker.get("code_predictor_config") or {})
    talker_is_root = False
    if not cp_cfg.get("hidden_size"):
        cp_cfg = dict(root.get("code_predictor_config") or {})
        talker = root
        talker_is_root = bool(cp_cfg.get("hidden_size"))
    if not cp_cfg.get("hidden_size"):
        logger.error(
            "code_predictor_config not found in %s/config.json (checked both "
            "talker_config.code_predictor_config and top-level "
            "code_predictor_config)", model_dir)
        sys.exit(1)
    key_prefix = ("code_predictor."
                  if talker_is_root else "talker.code_predictor.")
    cp_cfg["num_code_groups"] = talker.get("num_code_groups", 16)

    _export_sub_llm(
        model_dir,
        out_dir,
        model_class=Qwen3OmniNextCodePredictorCausalLM,
        sub_config=cp_cfg,
        key_prefix=key_prefix,
        model_type_override="qwen3_omni_next_code_predictor",
    )
    # ``torch.onnx.export`` drops ``axis=0`` from per-channel DequantizeLinear
    # nodes → TRT engine build fails with ``K == scaleSize``. Restore it.
    # No-op for the unquantized (FP16) CP.
    _patch_cp_dq_axis(os.path.join(out_dir, "model.onnx"))

    _extract_sidecars(model_dir,
                      out_dir,
                      _cp_codec_embed_and_head_specs(
                          key_prefix, talker.get("num_code_groups", 16)),
                      strict=True)
    # ``small_to_mtp_projection`` is fp32 in HF; C++ CodePredictor reads
    # it as __half. Downcast manually so the shared ``_to_fp16`` helper
    # keeps its bfloat16-only contract. Optional — some CP variants ship
    # without this projection (silent skip is intended).
    _write_downcast_fp16_sidecar(
        model_dir, out_dir, "small_to_mtp_projection.safetensors", [
            ("weight", f"{key_prefix}model.talker_projection.weight"),
            ("bias", f"{key_prefix}model.talker_projection.bias"),
        ])
    _patch_exported_config(out_dir,
                           root,
                           extra={
                               "use_embeddings_input":
                               True,
                               "num_code_groups":
                               talker.get("num_code_groups", 16),
                           })


def _export_talker(model_dir: str, llm_out_dir: str, model_type: str) -> None:
    """Export Talker LLM backbone + sidecar weights.

    The Talker is architecturally a dense Qwen3 CausalLM shared by Qwen3-TTS
    and Qwen3-Omni.  Per-family differences are confined to three places:

    - **Config source**: Qwen3-TTS reads the root ``config.json`` directly
      (root *is* the talker config).  Qwen3-Omni reads
      ``talker_config.text_config`` from the shared multi-stage root config.
    - **Key remap**: ``_talker_key_remap`` covers both — ``codec_embedding``
      → ``embed_tokens`` (both families) and ``codec_head`` → ``lm_head``
      (Qwen3-Omni only; no-op for Qwen3-TTS).
    - **Sidecars**: Qwen3-TTS writes ``text_embedding.safetensors`` +
      ``text_projection.safetensors``.  Qwen3-Omni writes ``embedding.safetensors``
      + ``hidden_projection.safetensors`` + ``text_projection.safetensors``
      (it takes thinker hidden states as input instead of a text embedding).
    """
    if model_type == "qwen3_omni_next":
        _export_omni_next_talker(model_dir, llm_out_dir)
        return

    from ..models.qwen3_tts import TalkerCausalLM

    is_omni = model_type in ("qwen3_omni", "qwen3_omni_moe")
    talker_cls = TalkerCausalLM
    if model_type == "qwen3_omni_moe":
        # The MoE Talker is a 128-expert MoE decoder, not the dense
        # CausalLM the TTS Talker class models.
        from ..models.qwen3_omni import Qwen3OmniMoeTalkerCausalLM
        talker_cls = Qwen3OmniMoeTalkerCausalLM
    _export_sub_llm(
        model_dir,
        llm_out_dir,
        model_class=talker_cls,
        sub_path=["talker_config", "text_config"] if is_omni else None,
        key_prefix="talker.",
        key_remap=_talker_key_remap,
    )

    logger.info("[Talker] Extracting weight sidecars ...")
    (_extract_omni_talker_sidecars if is_omni else _extract_tts_weights)(
        model_dir, llm_out_dir)

    logger.info("[Talker] Patching config.json with TTS fields ...")
    _patch_tts_config(model_dir, llm_out_dir)


# ---------------------------------------------------------------------------
# TTS CodePredictor export
# ---------------------------------------------------------------------------

_CP_RUNTIME_MODEL_TYPE = {
    "qwen3_tts": "qwen3_tts_code_predictor",
    "qwen3_omni": "qwen3_omni_moe_talker_code_predictor",
    "qwen3_omni_moe": "qwen3_omni_moe_talker_code_predictor",
}


def _export_code_predictor(model_dir: str, cp_out_dir: str,
                           model_type: str) -> None:
    """Export CodePredictor ONNX + extract codec embeddings / lm_heads / projection.

    The CodePredictor is a small 5-layer Qwen3 decoder shared by Qwen3-TTS
    and Qwen3-Omni.  Both families store the CP sub-config under
    ``talker_config.code_predictor_config`` in the root ``config.json``, so
    the extraction path is identical; only the runtime ``model_type`` string
    differs (used by the C++ runtime for identification).

    The CodePredictor has:
    - ``lm_heads`` + ``lm_head_idx`` as ONNX inputs (head gathered in-graph)
    - ``hidden_states`` as an additional output (for residual connection)
    - MLP FP16 overflow WAR applied to all layers

    Outputs:
    - ``model.onnx`` — CodePredictor ONNX graph
    - ``codec_embeddings.safetensors`` — 15 embedding tables
    - ``lm_heads.safetensors`` — 15 lm_head weights
    - ``small_to_mtp_projection.safetensors`` — talker→CP projection
    - ``config.json`` — LLM config with ``use_embeddings_input: true``
    """
    if model_type in ("qwen3_omni_next", "qwen3_omni_next_talker"):
        _export_omni_next_code_predictor(model_dir, cp_out_dir)
        return

    # ``code_predictor_config`` lives at either root.talker_config.* (full Omni
    # HF root) or root.* (Talker-only submodule export). Support both.
    root_config = _load_config(model_dir)
    talker_cfg = root_config.get("talker_config", {})
    cp_cfg = talker_cfg.get("code_predictor_config", {})
    talker_is_root = False
    if not cp_cfg.get("hidden_size"):
        cp_cfg = root_config.get("code_predictor_config", {})
        talker_cfg = root_config
        talker_is_root = bool(cp_cfg.get("hidden_size"))
    if not cp_cfg.get("hidden_size"):
        logger.error(
            "code_predictor_config not found in %s/config.json (checked both "
            "talker_config.code_predictor_config and top-level "
            "code_predictor_config)", model_dir)
        sys.exit(1)
    # Match either the full-Omni-root prefix or the Talker-only prefix.
    load_key_prefix = ("code_predictor."
                       if talker_is_root else "talker.code_predictor.")

    from ..models.qwen3_tts import CodePredictorCausalLM

    # onnx_export_spec needs the head count for the stacked lm_heads input.
    cp_cfg["num_code_groups"] = talker_cfg.get("num_code_groups", 16)

    _export_sub_llm(
        model_dir,
        cp_out_dir,
        model_class=CodePredictorCausalLM,
        sub_config=cp_cfg,
        key_prefix=load_key_prefix,
        model_type_override=_CP_RUNTIME_MODEL_TYPE.get(
            model_type, "qwen3_tts_code_predictor"),
    )

    # ``torch.onnx.export`` drops ``axis=0`` from per-channel DequantizeLinear
    # nodes → TRT engine build fails with ``K == scaleSize``. Restore it.
    _patch_cp_dq_axis(os.path.join(cp_out_dir, "model.onnx"))

    logger.info("[CodePredictor] Extracting weight files ...")
    _extract_code_predictor_weights(model_dir,
                                    cp_out_dir,
                                    talker_cfg,
                                    key_prefix=load_key_prefix)

    # Patch config.json with CodePredictor-specific fields.  Inline (not
    # ``_patch_exported_config``) to avoid the speaker_id normalisation, which
    # is Talker-only.
    cfg_path = os.path.join(cp_out_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["use_embeddings_input"] = True
        cfg["num_code_groups"] = talker_cfg.get("num_code_groups", 16)
        # CodePredictor has no deepstack visual inputs; override the
        # inherited value from ModelConfig.
        cfg["num_deepstack_features"] = 0
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)


def _cp_codec_embed_and_head_specs(prefix: str, num_code_groups: int) -> list:
    """Per-codebook codec_embeddings + lm_heads spec pair (shared by
    Qwen3-TTS, Qwen3-Omni, and Qwen3-Next Omni CodePredictors).
    """
    n = num_code_groups - 1  # first codebook lives in the Talker
    return [
        ("codec_embeddings.safetensors",
         [(f"embedding_{i}", f"{prefix}model.codec_embedding.{i}.weight")
          for i in range(n)], True),
        ("lm_heads.safetensors", [(f"lm_head_{i}.weight",
                                   f"{prefix}lm_head.{i}.weight")
                                  for i in range(n)], True),
    ]


def _patch_cp_dq_axis(onnx_path: str) -> None:
    """Restore ``axis=0`` on per-channel DequantizeLinear nodes.

    ``torch.onnx.export`` (dynamo, opset 24) silently drops the ``axis``
    attribute that ModelOpt configures via ``axis=0`` on the weight
    quantizer. The result is that every per-channel DQ node defaults to
    axis=1 at import time, and TRT then fails engine build with
    ``K == scaleSize`` because it interprets the scale vector along the
    wrong dim.

    Set ``axis=0`` on every DQ node whose scale initializer is 1-D and
    matches the FIRST dim of the weight initializer (ModelOpt's ``axis=0``
    convention for ``[out_features, in_features]`` layout). Per-tensor
    (scalar) DQ nodes are left alone.
    """
    import onnx
    from onnx import helper

    m = onnx.load(onnx_path, load_external_data=False)
    inits = {i.name: i for i in m.graph.initializer}
    patched = 0
    for n in m.graph.node:
        if n.op_type != "DequantizeLinear" or len(n.input) < 2:
            continue
        weight, scale = n.input[0], n.input[1]
        if weight not in inits or scale not in inits:
            continue
        s_dims = list(inits[scale].dims)
        if len(s_dims) == 0 or (len(s_dims) == 1 and s_dims[0] == 1):
            continue  # per-tensor
        if len(s_dims) != 1:
            continue
        w_dims = list(inits[weight].dims)
        if not w_dims or s_dims[0] != w_dims[0]:
            continue
        existing = [a for a in n.attribute if a.name == "axis"]
        if existing:
            if existing[0].i != 0:
                existing[0].i = 0
                patched += 1
            continue
        n.attribute.append(helper.make_attribute("axis", 0))
        patched += 1
    if patched:
        onnx.save(m,
                  onnx_path,
                  save_as_external_data=True,
                  all_tensors_to_one_file=True,
                  location=os.path.basename(onnx_path) + ".data",
                  size_threshold=1024)
        logger.info(
            "[CodePredictor] Patched axis=0 on %d DequantizeLinear nodes",
            patched)


def _extract_code_predictor_weights(model_dir: str, out_dir: str,
                                    talker_cfg: dict, key_prefix: str) -> None:
    """Extract codec_embeddings, lm_heads, and small_to_mtp_projection.

    ``key_prefix`` is either ``talker.code_predictor.`` (full-Omni HF root
    layout) or ``code_predictor.`` (Talker-only layout from older split
    exports).
    """
    from tensorrt_edgellm._safetensors_io import save_file

    weights = _load_all_weights(model_dir)

    # codec_embeddings: <prefix>model.codec_embedding.{i}.weight
    num_code_groups = talker_cfg.get("num_code_groups", 16)
    num_embeddings = num_code_groups - 1  # 15 for TTS (16-1=15)
    embedding_dict = {}
    for i in range(num_embeddings):
        key = f"{key_prefix}model.codec_embedding.{i}.weight"
        if key not in weights:
            logger.error("Key %r not found in checkpoint", key)
            sys.exit(1)
        embedding_dict[f"embedding_{i}"] = weights[key].cpu()
    save_file(_to_fp16(embedding_dict),
              os.path.join(out_dir, "codec_embeddings.safetensors"))
    logger.info(
        "[CodePredictor] Wrote codec_embeddings.safetensors "
        "(%d embeddings, shape %s)", num_embeddings,
        list(embedding_dict["embedding_0"].shape))

    # lm_heads: <prefix>lm_head.{i}.weight
    lm_head_dict = {}
    for i in range(num_embeddings):
        key = f"{key_prefix}lm_head.{i}.weight"
        if key not in weights:
            logger.error("Key %r not found in checkpoint", key)
            sys.exit(1)
        lm_head_dict[f"lm_head_{i}.weight"] = weights[key].cpu()
    save_file(_to_fp16(lm_head_dict),
              os.path.join(out_dir, "lm_heads.safetensors"))
    logger.info(
        "[CodePredictor] Wrote lm_heads.safetensors "
        "(%d heads, shape %s)", num_embeddings,
        list(lm_head_dict["lm_head_0.weight"].shape))

    # small_to_mtp_projection: <prefix>small_to_mtp_projection.{weight,bias}
    proj_w_key = f"{key_prefix}small_to_mtp_projection.weight"
    proj_b_key = f"{key_prefix}small_to_mtp_projection.bias"
    proj_dict = {}
    if proj_w_key in weights:
        proj_dict["weight"] = weights[proj_w_key].cpu()
        if proj_b_key in weights:
            proj_dict["bias"] = weights[proj_b_key].cpu()
        save_file(_to_fp16(proj_dict),
                  os.path.join(out_dir, "small_to_mtp_projection.safetensors"))
        logger.info(
            "[CodePredictor] Wrote small_to_mtp_projection.safetensors "
            "(weight shape %s)", list(proj_dict["weight"].shape))
    else:
        logger.warning("[CodePredictor] small_to_mtp_projection not found "
                       "(may be Omni-style without projection)")


# ---------------------------------------------------------------------------
# Action expert export (Alpamayo)
# ---------------------------------------------------------------------------


def _build_action_config(root_cfg: dict, weights: dict):
    """Build an ActionConfig from the Alpamayo root config and weight dict."""
    from .. import config as config_module

    expert_cfg = root_cfg.get("expert_cfg", {})
    head_dim = expert_cfg.get("head_dim", 128)

    # Infer num_hidden_layers by counting expert.layers.N keys.
    layer_indices = set()
    for k in weights:
        if k.startswith("expert.layers."):
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layer_indices.add(int(parts[2]))
    num_hidden_layers = len(layer_indices)

    # Infer num_key_value_heads from k_proj shape.
    num_kv_heads = expert_cfg.get("num_attention_heads", 0)
    for k, v in weights.items():
        if k.endswith("expert.layers.0.self_attn.k_proj.weight"):
            num_kv_heads = v.shape[0] // head_dim
            break

    traj_token_start_idx = root_cfg.get("traj_token_start_idx", 0)
    traj_cfg = root_cfg.get("traj_tokenizer_cfg", {})
    num_bins = traj_cfg.get("num_bins", 0)
    traj_token_start = traj_token_start_idx + num_bins

    in_proj_cfg = root_cfg.get("action_in_proj_cfg", {})

    return config_module.ActionConfig(
        rope_theta=5_000_000.0,
        mrope_section=[24, 20, 20],
        mrope_interleaved=True,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=expert_cfg.get("num_attention_heads", 0),
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        attention_scaling=config_module._get_attention_scaling(
            expert_cfg, head_dim, 1.0 / (float(head_dim)**0.5)),
        hidden_size=expert_cfg.get("hidden_size", 0),
        intermediate_size=expert_cfg.get("intermediate_size", 0),
        rms_norm_eps=1e-6,
        num_traj_tokens=1000,
        traj_token_start=traj_token_start,
        n_diffusion_tokens=root_cfg.get("action_space_cfg",
                                        {}).get("n_waypoints", 64),
        in_proj_hidden_size=in_proj_cfg.get("hidden_size", 512),
        in_proj_num_enc_layers=in_proj_cfg.get("num_enc_layers", 2),
        in_proj_max_freq=in_proj_cfg.get("max_freq", 100.0),
        in_proj_num_fourier_feats=in_proj_cfg.get("num_fourier_feats", 20),
    )


def _export_action(model_dir: str, action_out_dir: str, weights: dict,
                   config: dict, max_kv_cache_capacity: int,
                   dtype: "torch.dtype") -> None:
    """Export Alpamayo action expert to ONNX."""
    os.makedirs(action_out_dir, exist_ok=True)
    output_path = os.path.join(action_out_dir, "model.onnx")

    logger.info("[Action] Building ActionConfig from checkpoint ...")
    action_cfg = _build_action_config(config, weights)
    logger.info("[Action] Expert: %d layers, %d heads, hidden=%d",
                action_cfg.num_hidden_layers, action_cfg.num_attention_heads,
                action_cfg.hidden_size)

    logger.info("[Action] Exporting to %s", output_path)
    try:
        from ..onnx.export_encoder import (export_action_onnx,
                                           write_action_config)
        export_action_onnx(
            output_path=output_path,
            weights=weights,
            config=action_cfg,
            max_kv_cache_capacity=max_kv_cache_capacity,
            dtype=dtype,
        )
        write_action_config(action_cfg, max_kv_cache_capacity, action_out_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[Action] ONNX export failed")
        raise SystemExit(1) from exc
    logger.info("[Action] Done: %s", output_path)


def _prepare_alpamayo_visual_params(
    config: dict,
    weights: dict,
) -> "tuple[str, dict, dict]":
    """Return (vis_model_type, vis_config, vis_weights) for Alpamayo.

    Alpamayo uses a Qwen3-VL visual encoder.  This resolves the VLM config,
    remaps weight prefixes, and overrides vocab_size.
    """
    vlm_name = config.get("vlm_name_or_path", "Qwen/Qwen3-VL-8B-Instruct")
    vis_config = config
    if vlm_name:
        try:
            from transformers import AutoConfig
            vis_config = AutoConfig.from_pretrained(
                vlm_name, trust_remote_code=True).to_dict()
        except (ValueError, OSError) as exc:
            logger.warning(
                "[Visual] Failed to load VLM config from %s (%s); "
                "falling back to root config", vlm_name, exc)

    # Remap ``vlm.model.visual.*`` → ``model.visual.*``.
    vis_weights = {
        (k.replace("vlm.model.visual.", "model.visual.", 1) if k.startswith("vlm.model.visual.") else k):
        v
        for k, v in weights.items()
    }

    # Override vocab_size so the C++ runtime builds the correct embedding table.
    alpamayo_vocab = config.get("vocab_size")
    if alpamayo_vocab and alpamayo_vocab != vis_config.get("vocab_size"):
        vis_config["vocab_size"] = alpamayo_vocab
        _tc = vis_config.get("text_config")
        if isinstance(_tc, dict):
            _tc["vocab_size"] = alpamayo_vocab

    return vis_weights, vis_config, "qwen3_vl"


def _save_alpamayo_visual_processor(config: dict, visual_out_dir: str) -> None:
    """Save the Qwen3-VL processor with Alpamayo-specific pixel settings."""
    import shutil

    vlm_name = config.get("vlm_name_or_path", "Qwen/Qwen3-VL-8B-Instruct")
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(
            vlm_name,
            trust_remote_code=True,
            min_pixels=128 * 28 * 28,
            max_pixels=2048 * 32 * 32,
            size={
                "longest_edge": 16777216,
                "shortest_edge": 65536
            },
        )
        proc.save_pretrained(visual_out_dir)
        # Transformers v5 saves processor_config.json but the C++
        # runtime expects preprocessor_config.json.  Copy if needed.
        _proc_cfg = os.path.join(visual_out_dir, "processor_config.json")
        _pp_cfg = os.path.join(visual_out_dir, "preprocessor_config.json")
        if os.path.exists(_proc_cfg) and not os.path.exists(_pp_cfg):
            shutil.copy2(_proc_cfg, _pp_cfg)
        logger.info("[Visual] Saved Alpamayo processor sidecar files to %s",
                    visual_out_dir)
    except (ImportError, OSError, ValueError) as exc:
        logger.warning("[Visual] Failed to save Alpamayo processor: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Some onnx-library code paths create the `model.onnx.data` external-data
    # file via `open(path, 'wb')`, which applies the process umask to the
    # file mode. Container images that ship with a restrictive umask (0o077)
    # therefore produce 0o600 ONNX files that downstream engine-build hosts
    # cannot read when the ONNX directory is mounted as a different user.
    # Pin the umask to 0o022 so all exported artifacts come out world-readable.
    os.umask(0o022)

    p = argparse.ArgumentParser(
        prog="tensorrt-edgellm-export",
        description=(
            "Export ALL components of a multimodal checkpoint to ONNX "
            "(LLM + optional visual/audio encoder) in one command."),
    )
    p.add_argument(
        "model",
        help="Local checkpoint directory or Hugging Face model ID.",
    )
    p.add_argument(
        "output_dir",
        help=
        "Root output directory. Sub-dirs llm/, visual/, audio/ are created as needed.",
    )
    p.add_argument(
        "--dtype",
        default="float16",
        help="Weight dtype for visual/audio models (default: float16).",
    )
    p.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM backbone export (export only visual/audio encoders).",
    )
    p.add_argument(
        "--skip-visual",
        action="store_true",
        help="Skip visual encoder export.",
    )
    p.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip audio encoder export.",
    )
    p.add_argument(
        "--skip-code2wav",
        action="store_true",
        help="Skip Code2Wav vocoder export.",
    )
    p.add_argument(
        "--components",
        default="",
        help=
        ("Comma-separated allow-list of components to export. Default (empty) "
         "exports every component the checkpoint supports. Recognized values: "
         "thinker, mtp_draft, talker, code_predictor, visual, audio, "
         "code2wav, action; for Cosmos3 checkpoints: und_prefill, gen, "
         "vae_encoder. Useful for re-running a single stage, e.g. "
         "``--components code_predictor`` to refresh only the CodePredictor."),
    )
    p.add_argument(
        "--task",
        choices=("policy", "reasoning", "all"),
        default="all",
        help=
        ("Cosmos3-Edge checkpoints only: which task's artifacts to export. "
         "'policy' exports the und_prefill/gen/vae_encoder action-generation "
         "components, 'reasoning' exports the llm/ + visual/ backbones for "
         "the standard autoregressive VLM flow, 'all' (default) exports "
         "both."),
    )
    p.add_argument(
        "--action-chunk-size",
        "--action_chunk_size",
        dest="action_chunk_size",
        type=int,
        default=None,
        help=("Cosmos3 policy only: number of future action timesteps the GEN "
              "expert emits per request (action chunk length). Default "
              "(None) uses the checkpoint's canonical value (16)."),
    )
    p.add_argument(
        "--num-frames",
        "--num_frames",
        dest="num_frames",
        type=int,
        default=None,
        help=("Cosmos3 policy only: number of rollout frames the GEN/VAE "
              "components are shaped for (sets the VAE latent time axis). "
              "Default (None) uses the checkpoint's canonical value (17)."),
    )
    p.add_argument(
        "--fps",
        dest="fps",
        type=float,
        default=None,
        help=("Cosmos3 policy only: frames-per-second stamped into the GEN "
              "runtime config (controls the diffusion time schedule). Default "
              "(None) uses the checkpoint's canonical value (5)."),
    )
    p.add_argument(
        "--max-video-subsample-factor",
        "--max_video_subsample_factor",
        dest="max_video_subsample_factor",
        type=int,
        default=None,
        help=(
            "Cosmos3 policy only: largest video-subsample factor the GEN "
            "engine's DYNAMIC video-token profile must admit. The profile "
            "spans [latent_t(max_vsf) .. latent_t(1)]; a larger value widens "
            "the flexible range (e.g. 8 for finer subsampling) at the cost of "
            "a looser optimization profile. Default (None) uses 4."),
    )
    p.add_argument(
        "--min-action-chunk",
        "--min_action_chunk",
        dest="min_action_chunk",
        type=int,
        default=None,
        help=(
            "Cosmos3 policy only: smallest action-chunk length the GEN "
            "engine's DYNAMIC action-token axis must admit. Default (None) = "
            "the canonical chunk (action axis fixed). Set below the chunk to "
            "serve shorter action requests from one engine (e.g. 16)."),
    )
    p.add_argument(
        "--max-action-chunk",
        "--max_action_chunk",
        dest="max_action_chunk",
        type=int,
        default=None,
        help=(
            "Cosmos3 policy only: largest action-chunk length the GEN engine's "
            "DYNAMIC action-token axis must admit. Default (None) = the "
            "canonical chunk. Widen to serve longer action requests without a "
            "rebuild (keep it sane, e.g. <= 48, to bound tactic search)."),
    )
    p.add_argument(
        "--eagle-base",
        action="store_true",
        help=
        "Export as EAGLE3 base model (adds tree-attention I/O and hidden_states output).",
    )
    p.add_argument(
        "--eagle-draft-dir",
        default="",
        help=(
            "Path to the EAGLE3 draft checkpoint directory. Required for "
            "Gemma4 EAGLE3 base export so target hidden layers match the draft."
        ),
    )
    p.add_argument(
        "--fp8-embedding",
        "--fp8_embedding",
        dest="fp8_embedding",
        action="store_true",
        help=
        "Write embedding.safetensors in FP8 E4M3 format with per-row block scales.",
    )
    p.add_argument(
        "--reduced-vocab-dir",
        "--reduced_vocab_dir",
        dest="reduced_vocab_dir",
        default="",
        help=
        "Directory containing vocab_map.safetensors for LLM vocabulary reduction.",
    )
    p.add_argument(
        "--skip-softmax-scale-factor",
        "--skip_softmax_scale_factor",
        dest="skip_softmax_scale_factor",
        type=float,
        default=None,
        metavar="S",
        help=(
            "Skip-softmax (BLASST) calibrated scale factor S (0 = disabled). "
            "Baked into the AttentionPlugin nodes; at inference the runtime "
            "derives lambda = S / context_length per request for the prefill "
            "FMHA. Obtain S from calibrate_skip_softmax.py. Overrides the "
            "checkpoint config.json key \"skip_softmax_scale_factor\" — an "
            "explicit 0 disables skip-softmax even if the config enables it; "
            "omit the flag to keep the config value."),
    )
    p.add_argument(
        "--draft-reduced-vocab-dir",
        dest="draft_reduced_vocab_dir",
        default="",
        metavar="DIR",
        help=
        ("Directory containing vocab_map.safetensors for the DFlash draft model "
         "(from tensorrt_edgellm/scripts/reduce_vocab.py). "
         "Reduces the DFlash draft lm_head output dimension."),
    )
    p.add_argument(
        "--mtp",
        action="store_true",
        help=
        ("Export MTP components. Qwen-style checkpoints use checkpoint-internal "
         "MTP weights; paired MTP models such as Gemma4 require --mtp-draft-dir."
         ),
    )
    p.add_argument(
        "--mtp-draft-dir",
        "--mtp_draft_dir",
        dest="mtp_draft_dir",
        default="",
        help=(
            "Path to a paired MTP draft checkpoint. Required for Gemma4 MTP; "
            "not used by Qwen-style checkpoint-internal MTP."),
    )
    p.add_argument(
        "--mtpDraftModelDir",
        dest="mtp_draft_dir",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--gemma4-mtp-assistant-dir",
        default="",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--mtp-tree-base",
        action="store_true",
        help="Export MTP base with DDTree hybrid state metadata inputs "
        "(implies --mtp; required for MTP tree drafting).",
    )
    p.add_argument(
        "--dflash-base",
        action="store_true",
        help="Export as DFlash base model (adds DFlash hidden_states output).",
    )
    p.add_argument(
        "--dflash-tree-base",
        action="store_true",
        help="Export DFlash base with DDTree hybrid state metadata inputs.",
    )
    p.add_argument(
        "--dflash-draft",
        action="store_true",
        help="Export DFlash draft model.",
    )
    p.add_argument(
        "--dflash-draft-dir",
        default="",
        help="Path to the DFlash draft checkpoint directory.",
    )
    p.add_argument(
        "--jetspec-base",
        action="store_true",
        help="Export as JetSpec base model (adds target hidden-state output).",
    )
    p.add_argument(
        "--jetspec-tree-base",
        action="store_true",
        help="Export JetSpec base with DDTree hybrid state metadata inputs.",
    )
    p.add_argument(
        "--jetspec-draft",
        action="store_true",
        help="Export JetSpec draft model.",
    )
    p.add_argument(
        "--jetspec-draft-dir",
        default="",
        help="Path to the JetSpec draft checkpoint directory.",
    )
    p.add_argument(
        "--dspark-base",
        action="store_true",
        help="Export as DSpark base model (adds target hidden-state output).",
    )
    p.add_argument(
        "--dspark-draft",
        action="store_true",
        help=
        "Export DSpark draft backbone model plus Markov/confidence sidecars.",
    )
    p.add_argument(
        "--dspark-draft-dir",
        default="",
        help="Path to the DSpark draft checkpoint directory.",
    )
    p.add_argument(
        "--externalize-weights",
        nargs="+",
        choices=EXTERNAL_WEIGHT_CHOICES,
        default=[],
        metavar="WEIGHT_TYPE",
        help=("Expose selected model weights as ONNX inputs and write them "
              "to safetensors external weight files. Values: int4_ffn, "
              "int4_moe, nvfp4_moe, lm_head, all."),
    )
    p.add_argument(
        "--max-kv-cache-capacity",
        type=int,
        default=4096,
        help=
        "Max KV cache capacity for action expert (Alpamayo). Default: 4096.",
    )
    p.add_argument(
        "--skip-action",
        action="store_true",
        help="Skip action expert export (Alpamayo).",
    )
    p.add_argument(
        "--tp-size",
        "--tp_size",
        dest="tp_size",
        type=int,
        default=1,
        help=
        ("Tensor-parallel world size (default: 1 = single device). "
         "When >1, exports per-rank LLM ONNX files named model_world{N}_rank{R}.onnx."
         ),
    )
    p.add_argument(
        "--talker-sidecar-from",
        "--talker_sidecar_from",
        dest="talker_sidecar_from",
        default="",
        help=(
            "HF root checkpoint from which to extract the Qwen3-Omni Talker "
            "sidecars (hidden_projection, text_projection, codec embedding). "
            "Used when exporting a standalone NVFP4 Talker checkpoint whose "
            "model.safetensors omits these projection weights."),
    )
    p.add_argument(
        "--num-decoder-layer",
        "--num_decoder_layer",
        dest="num_decoder_layer",
        type=int,
        default=None,
        help=(
            "Accuracy-debugging only: export only the first N decoder layers "
            "of the LLM backbone. The runtime config.json and the ONNX KV "
            "in/out count follow N automatically. Supported for the plain "
            "default model path (e.g. Qwen3) and hybrid base models (e.g. "
            "Qwen3.5, Nemotron-H); not for eagle/mtp/dflash/jetspec "
            "speculative-decoding variants."),
    )
    p.add_argument(
        "--int4-gemm-plugin-version",
        "--int4_gemm_plugin_version",
        dest="int4_gemm_plugin_version",
        type=int,
        choices=[1, 2],
        default=2,
        help=("INT4 groupwise GEMM plugin backend to export with. "
              "2 (default) targets the cuteDSL Int4GroupwiseGemmPluginV2 with "
              "fragment-layout weights; 1 targets the legacy "
              "Int4GroupwiseGemmPlugin with AWQ-swizzled weights."),
    )
    p.add_argument(
        "--quantization",
        default=None,
        choices=["int4_awq", "nvfp4"],
        help=("Override quantization type for BF16/FP16 checkpoints. "
              "Applies on-the-fly quantization during export (e.g. INT4 RTN "
              "for QAT models stored in BF16)."),
    )
    args = p.parse_args()

    # Select the INT4 GEMM plugin backend before any weight repack / op emission.
    set_int4_gemm_plugin_version(args.int4_gemm_plugin_version)

    model_dir = _resolve_model_dir(args.model)
    config = _load_config(model_dir)
    model_type: str = config.get("model_type", "unknown")
    dtype = _dtype_from_str(args.dtype)

    # Cosmos3-Edge checkpoints carry two model families that run on DIFFERENT
    # runtime paths; ``--task`` selects which artifact set this invocation
    # exports (both by default):
    #   * policy    -> und_prefill/gen/vae_encoder components for the
    #     experimental component runtime, and
    #   * reasoning -> a regular llm/ backbone + visual/ SigLIP2 encoder for
    #     the standard llm_build + visual_build + llm_inference VLM flow.
    # Both the root ``model_type`` and the diffusers ``model_index.json``
    # identify them.
    if model_type in ("cosmos3_edge",
                      "cosmos3_omni") or _is_cosmos3_checkpoint(model_dir):
        has_reasoner = model_type == "cosmos3_edge"
        if args.task == "reasoning" and not has_reasoner:
            p.error("--task reasoning requires a cosmos3_edge checkpoint "
                    "(this checkpoint carries no reasoner tower)")

        if args.task in ("policy", "all"):
            from ..models.cosmos3.export import export_cosmos3_components
            requested = [c for c in args.components.split(",") if c] or None
            # Forward only the variables the user set; leaving one unset keeps
            # the module's canonical default (chunk=16, num_frames=17, fps=5).
            policy_overrides = {
                k: v
                for k, v in (("action_chunk_size", args.action_chunk_size),
                             ("num_frames", args.num_frames), ("fps",
                                                               args.fps),
                             ("max_video_subsample_factor",
                              args.max_video_subsample_factor),
                             ("min_action_chunk", args.min_action_chunk),
                             ("max_action_chunk", args.max_action_chunk))
                if v is not None
            }
            export_cosmos3_components(model_dir,
                                      args.output_dir,
                                      components=requested,
                                      dtype=dtype,
                                      **policy_overrides)

        if args.task in ("reasoning", "all") and has_reasoner:
            # Text decoder -> regular llm/ backbone (KV-cache autoregressive
            # decode via the standard runtime). Cosmos3ReasonerCausalLM is
            # registered for "cosmos3_edge"/"cosmos3_edge_text" in the package
            # __init__ like every other model family.
            if not args.skip_llm:
                _export_llm(model_dir,
                            os.path.join(args.output_dir, "llm"),
                            model_type="cosmos3_edge")
            # SigLIP2 ViT + PatchMerger -> visual/ for the standard
            # visual_build + multimodal runtime. The vision tower is read
            # directly from its checkpoint shards (the root index maps it to
            # per-component files, so there is no flat *.safetensors set).
            if not args.skip_visual:
                from ..model import load_model_config
                from ..models.cosmos3_reasoner import \
                    load_cosmos3_reasoner_visual_checkpoint
                _export_visual(
                    model_dir,
                    os.path.join(args.output_dir, "visual"),
                    load_cosmos3_reasoner_visual_checkpoint(model_dir),
                    config,
                    "cosmos3_edge",
                    dtype,
                    model_config=load_model_config(model_dir))
        return

    has_mtp_draft = _has_mtp(config)
    is_gemma4_target = model_type in _GEMMA4_MODEL_TYPES
    mtp_draft_dir_arg = args.mtp_draft_dir or args.gemma4_mtp_assistant_dir
    gemma4_mtp_requested = args.mtp and is_gemma4_target
    gemma4_mtp_assistant_dir = ""
    gemma4_kv_sharing_map: list[dict] = []
    externalize_weights = resolve_externalize_weights(args.externalize_weights)

    if (model_type == "qwen3_tts" and config.get("tts_model_type")
            not in ("custom_voice", "voice_design", "base")):
        p.error(
            "Only Qwen3-TTS CustomVoice / VoiceDesign / Base checkpoints are "
            f"supported. Got tts_model_type={config.get('tts_model_type')!r}.")

    if args.mtp_tree_base:
        args.mtp = True
    if args.tp_size > 1 and (args.eagle_base or args.mtp or args.dflash_base
                             or args.dflash_tree_base or args.dflash_draft
                             or args.dspark_base or args.dspark_draft
                             or gemma4_mtp_requested):
        p.error(
            "Tensor-parallel speculative decoding export is not supported.")
    if args.tp_size > 1 and externalize_weights:
        p.error(
            "--externalize-weights is not supported with --tp-size > 1 because its weight files and manifest "
            "are not rank-local yet")

    if args.eagle_base and args.mtp:
        p.error("--eagle-base and --mtp cannot be enabled together")
    if args.eagle_draft_dir and not args.eagle_base:
        p.error("--eagle-draft-dir requires --eagle-base")
    if args.eagle_base and is_gemma4_target and not args.eagle_draft_dir:
        p.error("Gemma4 --eagle-base requires --eagle-draft-dir")
    if args.mtp_draft_dir and args.gemma4_mtp_assistant_dir:
        p.error("Use only one MTP draft checkpoint directory option")
    if mtp_draft_dir_arg and args.eagle_base:
        p.error("--mtp-draft-dir cannot be combined with --eagle-base")
    if args.dflash_tree_base:
        args.dflash_base = True
    if args.jetspec_tree_base:
        args.jetspec_base = True
    if mtp_draft_dir_arg and (args.dflash_base or args.dflash_draft
                              or args.jetspec_base or args.jetspec_draft
                              or args.dspark_base or args.dspark_draft):
        p.error("--mtp-draft-dir cannot be combined with "
                "DFlash/JetSpec/DSpark export")
    if args.dflash_base and (args.eagle_base or args.mtp or args.jetspec_base
                             or args.jetspec_draft or args.dspark_base
                             or args.dspark_draft):
        p.error("--dflash-base cannot be combined with "
                "EAGLE/MTP/JetSpec/DSpark modes")
    if args.dflash_draft and (args.eagle_base or args.mtp or args.jetspec_base
                              or args.jetspec_draft or args.dspark_base
                              or args.dspark_draft):
        p.error("--dflash-draft cannot be combined with "
                "EAGLE/MTP/JetSpec/DSpark modes")
    if args.dflash_draft and not args.dflash_draft_dir:
        p.error("--dflash-draft requires --dflash-draft-dir")
    if args.jetspec_base and (args.eagle_base or args.mtp or args.dflash_base
                              or args.dflash_draft or args.dspark_base
                              or args.dspark_draft):
        p.error("--jetspec-base cannot be combined with "
                "EAGLE/MTP/DFlash/DSpark modes")
    if args.jetspec_draft and (args.eagle_base or args.mtp or args.dflash_base
                               or args.dflash_draft or args.dspark_base
                               or args.dspark_draft):
        p.error("--jetspec-draft cannot be combined with "
                "EAGLE/MTP/DFlash/DSpark modes")
    if args.jetspec_base and not args.jetspec_draft_dir:
        p.error("--jetspec-base requires --jetspec-draft-dir "
                "for target layer metadata")
    if args.jetspec_draft and not args.jetspec_draft_dir:
        p.error("--jetspec-draft requires --jetspec-draft-dir")
    if args.dspark_base and (args.eagle_base or args.mtp or args.dflash_base
                             or args.dflash_draft or args.jetspec_base
                             or args.jetspec_draft):
        p.error("--dspark-base cannot be combined with "
                "EAGLE/MTP/DFlash/JetSpec modes")
    if args.dspark_draft and (args.eagle_base or args.mtp or args.dflash_base
                              or args.dflash_draft or args.jetspec_base
                              or args.jetspec_draft):
        p.error("--dspark-draft cannot be combined with "
                "EAGLE/MTP/DFlash/JetSpec modes")
    if args.dspark_base and not args.dspark_draft_dir:
        p.error("--dspark-base requires --dspark-draft-dir "
                "for target layer metadata")
    if args.dspark_draft and not args.dspark_draft_dir:
        p.error("--dspark-draft requires --dspark-draft-dir")
    if args.mtp and args.skip_llm:
        p.error("--mtp requires LLM export; remove --skip-llm")
    if mtp_draft_dir_arg and args.skip_llm:
        p.error("--mtp-draft-dir requires LLM export; remove --skip-llm")
    if args.dflash_base and args.skip_llm:
        p.error("--dflash-base requires LLM export; remove --skip-llm")
    if args.jetspec_base and args.skip_llm:
        p.error("--jetspec-base requires LLM export; remove --skip-llm")
    if args.dspark_base and args.skip_llm:
        p.error("--dspark-base requires LLM export; remove --skip-llm")
    if args.dflash_draft and args.skip_llm:
        logger.info(
            "--dflash-draft implies --skip-llm (draft export is independent)")
    if args.jetspec_draft and args.skip_llm:
        logger.info(
            "--jetspec-draft implies --skip-llm (draft export is independent)")
    if args.dspark_draft and args.skip_llm:
        logger.info(
            "--dspark-draft implies --skip-llm (draft export is independent)")
    if mtp_draft_dir_arg and not args.mtp:
        p.error("--mtp-draft-dir requires --mtp")
    if args.mtp and is_gemma4_target and not mtp_draft_dir_arg:
        p.error("Gemma4 --mtp requires --mtp-draft-dir <assistant checkpoint>")
    if args.mtp and not is_gemma4_target and mtp_draft_dir_arg:
        p.error("--mtp-draft-dir is currently only supported for Gemma4 MTP")
    if args.mtp and not is_gemma4_target and not has_mtp_draft:
        p.error("--mtp was requested, but the checkpoint does not expose "
                "MTP weights/config")
    if gemma4_mtp_requested:
        try:
            gemma4_mtp_assistant_dir = _resolve_mtp_draft_dir(
                mtp_draft_dir_arg)
            if not os.path.isdir(gemma4_mtp_assistant_dir):
                p.error("Gemma4 MTP draft directory not found: %s" %
                        gemma4_mtp_assistant_dir)
            gemma4_kv_sharing_map = _validate_gemma4_mtp_pair(
                model_dir, gemma4_mtp_assistant_dir)
        except ValueError as exc:
            p.error(str(exc))
    if args.num_decoder_layer is not None:
        if args.num_decoder_layer < 1:
            p.error("--num-decoder-layer must be >= 1")
        if (args.eagle_base or args.mtp or args.dflash_base
                or args.dflash_draft or args.jetspec_base or args.jetspec_draft
                or args.dspark_base or args.dspark_draft):
            p.error("--num-decoder-layer cannot be combined with "
                    "--eagle-base / --mtp / --dflash-base / --dflash-draft / "
                    "--jetspec-base / --jetspec-draft / "
                    "--dspark-base / --dspark-draft")

    _VALID_COMPONENTS = {
        "thinker", "mtp_draft", "dflash_draft", "jetspec_draft",
        "dspark_draft", "talker", "code_predictor", "visual", "audio",
        "code2wav", "action", "dllm"
    }
    requested_components = {
        c.strip()
        for c in args.components.split(",") if c.strip()
    }
    unknown = requested_components - _VALID_COMPONENTS
    if unknown:
        p.error(f"--components contains unknown values {sorted(unknown)}; "
                f"valid choices: {sorted(_VALID_COMPONENTS)}")

    if _is_diffusion_gemma(model_type, config):
        has_diffusion_visual = _diffusion_gemma_has_visual(config)
        allowed = {
            "thinker",
            "dllm",
        }
        if has_diffusion_visual:
            allowed.add("visual")
        disallowed = requested_components - allowed
        if disallowed:
            p.error("DiffusionGemma supports only components "
                    f"{sorted(allowed)}; got {sorted(disallowed)}")
        wants_diffusion_engines = (not args.skip_llm
                                   and (not requested_components
                                        or bool(requested_components & {
                                            "thinker",
                                            "dllm",
                                        })))
        wants_visual = (has_diffusion_visual and not args.skip_visual
                        and (not requested_components
                             or "visual" in requested_components))
        if args.skip_llm and not wants_visual:
            p.error("DiffusionGemma --skip-llm is only valid when exporting "
                    "the visual component.")
        if not wants_diffusion_engines and not wants_visual:
            p.error("No DiffusionGemma components selected for export.")
        if (args.mtp or args.eagle_base or args.dflash_base
                or args.dflash_draft or args.jetspec_base or args.jetspec_draft
                or args.dspark_base or args.dspark_draft):
            p.error("DiffusionGemma cannot be combined with speculative "
                    "decode export flags")
        if args.reduced_vocab_dir:
            p.error("DiffusionGemma unified self-conditioning does not "
                    "support --reduced-vocab-dir because hidden feedback "
                    "must align with the full embedding table.")

        logger.info("=" * 60)
        logger.info("Model type    : diffusion_gemma_text")
        logger.info("Checkpoint    : %s", model_dir)
        logger.info("Output dir    : %s", args.output_dir)
        logger.info("  %-15s: %s", "dllm",
                    "yes" if wants_diffusion_engines else "no")
        logger.info("  %-15s: %s", "visual", "yes" if wants_visual else "no")
        logger.info("FP8 embedding : %s",
                    "yes" if args.fp8_embedding else "no")
        logger.info("TP size       : %d", args.tp_size)
        logger.info("=" * 60)
        if wants_diffusion_engines:
            _export_diffusion_gemma(
                model_dir,
                args.output_dir,
                fp8_embedding=args.fp8_embedding,
                reduced_vocab_dir=args.reduced_vocab_dir,
                externalize_weights=externalize_weights,
                tp_size=args.tp_size,
            )
        if wants_visual:
            from ..model import load_model_config
            weights = _load_all_weights(model_dir)
            model_config = load_model_config(model_dir)
            _export_diffusion_gemma_visual(
                model_dir,
                os.path.join(args.output_dir, "visual"),
                weights,
                config,
                dtype,
                model_config=model_config,
            )

        print()
        print("=" * 60)
        print("Export complete")
        print(f"  output dir: {args.output_dir}")
        for component in ("dllm", "visual"):
            p_sub = os.path.join(args.output_dir, component)
            onnx = os.path.join(p_sub, "model.onnx")
            mb = os.path.getsize(onnx) / 1e6 if os.path.exists(onnx) else 0
            if os.path.exists(onnx):
                print(f"  {component:27s}: {onnx}  ({mb:.1f} MB)")
                for sidecar in ("external_nvfp4_moe_weights.safetensors", ):
                    sc_path = os.path.join(p_sub, sidecar)
                    if os.path.exists(sc_path):
                        sc_mb = os.path.getsize(sc_path) / 1e6
                        print(f"                             + {sidecar}  "
                              f"({sc_mb:.1f} MB)")
        print("=" * 60)
        return

    # Load weights lazily — only needed when a weight-consuming exporter runs.
    _weights: dict = {}

    def _get_weights() -> dict:
        nonlocal _weights
        if not _weights:
            logger.info("Loading safetensors weights ...")
            _weights.update(_load_all_weights(model_dir))
        return _weights

    # Parse ModelConfig lazily so quantized visual towers get the right
    # Linear dispatch through make_linear.  Non-quantized checkpoints simply
    # produce a ModelConfig with quant_type=fp16, so this never silently
    # fails — a real exception means the checkpoint is malformed and we
    # should surface it.
    _model_config: "list[Optional[ModelConfig]]" = [None]

    def _get_model_config() -> "ModelConfig":
        if _model_config[0] is None:
            from ..model import load_model_config
            _model_config[0] = load_model_config(model_dir)
        return _model_config[0]

    def _get_code2wav_weights() -> dict:
        if model_type == "qwen3_tts":
            return {}
        return _get_weights()

    # When only a standalone draft flag is set, run just that draft stage.
    # If a matching base flag is also set, export both base and draft artifacts.
    _draft_only = ((args.dflash_draft and not args.dflash_base)
                   or (args.jetspec_draft and not args.jetspec_base)
                   or (args.dspark_draft and not args.dspark_base))

    def _export_visual_component(out: str) -> None:
        if _is_alpamayo(model_type):
            _export_alpamayo_visual(model_dir,
                                    out,
                                    _get_weights(),
                                    config,
                                    dtype,
                                    model_config=_get_model_config())
            return
        _export_visual(model_dir,
                       out,
                       _get_weights(),
                       config,
                       model_type,
                       dtype,
                       model_config=_get_model_config())

    # `--components` is a per-component allow-list. An empty list (the default)
    # means "no restriction": every component the checkpoint supports runs.
    def _allow(component: str) -> bool:
        return not requested_components or component in requested_components

    # Each stage is (enabled, component_name, exporter_callable). Exporter
    # receives the computed output dir; the (enabled, component) columns also
    # drive both the pre-run log and the post-run summary below.
    stages = [
        (_has_llm_component(model_type, "thinker") and not args.skip_llm
         and not _draft_only
         and _allow("thinker"), "thinker", lambda out: _export_llm(
             model_dir,
             out,
             model_type=model_type,
             eagle_base=args.eagle_base,
             eagle_draft_dir=args.eagle_draft_dir,
             mtp_base=args.mtp and not gemma4_mtp_requested,
             mtp_tree_base=args.mtp_tree_base,
             dflash_base=args.dflash_base,
             dflash_tree_base=args.dflash_tree_base,
             dflash_draft_dir=args.dflash_draft_dir,
             jetspec_base=args.jetspec_base,
             jetspec_tree_base=args.jetspec_tree_base,
             jetspec_draft_dir=args.jetspec_draft_dir,
             dspark_base=args.dspark_base,
             dspark_draft_dir=args.dspark_draft_dir,
             gemma4_mtp_base=gemma4_mtp_requested,
             fp8_embedding=args.fp8_embedding,
             reduced_vocab_dir=args.reduced_vocab_dir,
             externalize_weights=externalize_weights,
             tp_size=args.tp_size,
             num_decoder_layers=args.num_decoder_layer,
             skip_softmax_scale_factor=args.skip_softmax_scale_factor,
             quantization_override=getattr(args, 'quantization', None))),
        (args.mtp and not gemma4_mtp_requested
         and _allow("mtp_draft"), "mtp_draft", lambda out: _export_mtp_draft(
             model_dir, out, externalize_weights=externalize_weights)),
        (gemma4_mtp_requested and _allow("mtp_draft"),
         "mtp_draft", lambda out: _export_gemma4_mtp_draft(
             model_dir, out, gemma4_mtp_assistant_dir, gemma4_kv_sharing_map)),
        (args.dflash_draft, "dflash_draft", lambda out: _export_dflash_draft(
            model_dir,
            out,
            args.dflash_draft_dir,
            draft_reduced_vocab_dir=args.draft_reduced_vocab_dir)),
        (args.jetspec_draft, "jetspec_draft",
         lambda out: _export_jetspec_draft(model_dir,
                                           out,
                                           args.jetspec_draft_dir,
                                           draft_reduced_vocab_dir=args.
                                           draft_reduced_vocab_dir)),
        (args.dspark_draft, "dspark_draft", lambda out: _export_dspark_draft(
            model_dir, out, args.dspark_draft_dir)),
        (_has_llm_component(model_type, "talker") and not args.skip_llm
         and not _draft_only and _allow("talker"), "talker",
         lambda out: _export_talker(model_dir, out, model_type)),
        (_has_llm_component(model_type, "code_predictor") and not args.skip_llm
         and not _draft_only and _allow("code_predictor"), "code_predictor",
         lambda out: _export_code_predictor(model_dir, out, model_type)),
        (_has_visual(model_type) and not args.skip_visual and not _draft_only
         and _allow("visual"), "visual", _export_visual_component),
        (
            _has_audio(model_type) and not args.skip_audio and not _draft_only
            and _checkpoint_audio_config(config) is not None
            and _allow("audio"),
            "audio",
            lambda out: _export_audio(
                model_dir,
                out,
                _get_weights(),
                config,
                model_type,
                dtype,
                # Nemotron-3.5-ASR has no LLM backbone, so the LLM-oriented
                # ModelConfig (which requires a top-level ``hidden_size``) does
                # not apply; its fp16 encoder does not need it.
                model_config=(None if model_type == "nemotron3_5_asr" else
                              _get_model_config()))),
        (_has_rnnt_decoder(model_type) and not args.skip_audio
         and not _draft_only and _checkpoint_audio_config(config) is not None
         and _allow("rnnt_decoder"), "rnnt_decoder", lambda out:
         _export_rnnt_decoder(model_dir, out, _get_weights(), config, dtype)),
        (_has_code2wav(model_type) and not args.skip_code2wav
         and not _draft_only and _allow("code2wav"), "code2wav",
         lambda out: _export_code2wav(model_dir, out, _get_code2wav_weights(),
                                      config, model_type, dtype)),
        (_has_action(model_type) and not args.skip_action and not _draft_only
         and _allow("action"), "action", lambda out: _export_action(
             model_dir,
             out,
             _get_weights(),
             config,
             max_kv_cache_capacity=args.max_kv_cache_capacity,
             dtype=dtype))
    ]

    logger.info("=" * 60)
    logger.info("Model type    : %s", model_type)
    logger.info("Checkpoint    : %s", model_dir)
    logger.info("Output dir    : %s", args.output_dir)
    for enabled, component, _ in stages:
        logger.info("  %-15s: %s", component, "yes" if enabled else "no")
    logger.info("FP8 embedding : %s", "yes" if args.fp8_embedding else "no")
    logger.info("MTP capable   : %s", "yes" if has_mtp_draft else "no")
    logger.info("MTP export    : %s",
                "yes" if args.mtp or gemma4_mtp_requested else "no")
    logger.info("Gemma4 MTP    : %s",
                gemma4_mtp_assistant_dir if gemma4_mtp_assistant_dir else "no")
    logger.info("DFlash base   : %s", "yes" if args.dflash_base else "no")
    logger.info("DFlash draft  : %s", "yes" if args.dflash_draft else "no")
    logger.info("JetSpec base  : %s", "yes" if args.jetspec_base else "no")
    logger.info("JetSpec draft : %s", "yes" if args.jetspec_draft else "no")
    logger.info("DSpark base   : %s", "yes" if args.dspark_base else "no")
    logger.info("DSpark draft  : %s", "yes" if args.dspark_draft else "no")
    logger.info("Reduced vocab : %s",
                args.reduced_vocab_dir if args.reduced_vocab_dir else "no")
    logger.info(
        "External weights: %s",
        ", ".join(externalize_weights) if externalize_weights else "no")
    logger.info("TP size       : %d", args.tp_size)
    if args.num_decoder_layer is not None:
        logger.info("Decoder layers: first %d only (accuracy debug)",
                    args.num_decoder_layer)
    logger.info("=" * 60)

    # ``--fp8-embedding`` only applies to the LLM thinker.  Models without a
    # thinker (e.g. Qwen3-TTS) silently fall back to FP16 — warn so the user
    # isn't surprised when the flag has no effect.
    if args.fp8_embedding and not _has_llm_component(model_type, "thinker"):
        logger.warning(
            "--fp8-embedding is not supported for Talker / CodePredictor; "
            "using FP16 embeddings.")

    if (_has_audio(model_type) and not args.skip_audio
            and _checkpoint_audio_config(config) is None):
        logger.warning(
            "Model type '%s' supports audio, but this checkpoint has no "
            "audio_config — skipping audio encoder export.", model_type)

    for enabled, component, fn in stages:
        if enabled:
            fn(
                os.path.join(args.output_dir,
                             _layout_for(model_type, component)))

    # Standalone NVFP4 Qwen3-Omni-MoE Talker checkpoints (model_type=
    # ``qwen3_omni_moe_talker``) ship only the LLM backbone weights and
    # codec_embedding; the hidden_projection / text_projection MLP weights
    # live in the original HF root checkpoint. When ``--talker-sidecar-from``
    # points at that HF root, extract those sidecars into the talker output
    # so the C++ runtime can locate them next to the engine.
    if args.talker_sidecar_from and model_type == "qwen3_omni_moe_talker":
        talker_out = os.path.join(args.output_dir,
                                  _layout_for(model_type, "thinker"))
        if os.path.isdir(talker_out):
            logger.info("[Talker-Omni] Extracting sidecars from %s into %s",
                        args.talker_sidecar_from, talker_out)
            _extract_omni_talker_sidecars(args.talker_sidecar_from, talker_out)

    # Summary
    _SIDECARS = ("embedding.safetensors", "ple_embedding.safetensors",
                 "text_embedding.safetensors", "text_projection.safetensors",
                 "hidden_projection.safetensors",
                 "codec_embedding.safetensors", "codec_embeddings.safetensors",
                 "lm_heads.safetensors",
                 "speaker_codec_embeddings.safetensors",
                 "small_to_mtp_projection.safetensors",
                 "external_int4_ffn_weights.safetensors",
                 "external_int4_moe_weights.safetensors",
                 "external_lm_head_weight.safetensors")
    print()
    print("=" * 60)
    print("Export complete")
    print(f"  output dir: {args.output_dir}")
    for _, component, _ in stages:
        p_sub = os.path.join(args.output_dir,
                             _layout_for(model_type, component))
        if not os.path.isdir(p_sub):
            continue
        onnx = os.path.join(p_sub, "model.onnx")
        mb = os.path.getsize(onnx) / 1e6 if os.path.exists(onnx) else 0
        print(f"  {component:15s}: {onnx}  ({mb:.1f} MB)")
        for sidecar in _SIDECARS:
            sc_path = os.path.join(p_sub, sidecar)
            if os.path.exists(sc_path):
                sc_mb = os.path.getsize(sc_path) / 1e6
                print(f"                   + {sidecar}  ({sc_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
