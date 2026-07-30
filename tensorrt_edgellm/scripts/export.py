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

from ..checkpoint.checkpoint_utils import normalize_rope_scaling_for_runtime
from ..external_weights import (EXTERNAL_WEIGHT_CHOICES,
                                resolve_externalize_weights)

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

_VLM_MODEL_TYPES = frozenset([
    "qwen3_vl",
    "qwen3_omni",
    "qwen3_omni_moe",
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
    "gemma4",
    "qwen3_asr",
    "qwen3_omni",
    "qwen3_omni_thinker",
    "qwen3_omni_moe",
    "qwen3_omni_moe_thinker",
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
_ASR_LLM_MODEL_TYPES = (_AUDIO_MODEL_TYPES - _NEMOTRON_OMNI_MODEL_TYPES -
                        {"gemma4_unified"})

_CODE2WAV_MODEL_TYPES = frozenset([
    "qwen3_omni",
    "qwen3_omni_moe",
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
}
_DEFAULT_LLM_COMPONENTS = frozenset(["thinker"])


def _has_visual(model_type: str) -> bool:
    return model_type in _VLM_MODEL_TYPES


def _has_audio(model_type: str) -> bool:
    return model_type in _AUDIO_MODEL_TYPES


def _checkpoint_audio_config(config: dict) -> "dict | None":
    """Locate the audio-encoder config wherever the checkpoint stores it.

    Gemma4 / Gemma4-Unified keep ``audio_config`` at the root; Qwen3-ASR /
    Qwen3-Omni nest it under ``thinker_config``; Nemotron-Omni names it
    ``sound_config``. Returns ``None`` when the checkpoint genuinely has no
    audio encoder (e.g. Gemma4 dense with ``"audio_config": null``).
    """
    return (config.get("audio_config")
            or (config.get("thinker_config") or {}).get("audio_config")
            or config.get("sound_config"))


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


# Default output sub-path for every component.  Model types that need a
# non-default path only list the components that differ in ``_LAYOUT_OVERRIDES``.
_DEFAULT_LAYOUT: dict[str, str] = {
    "thinker": "llm",
    "talker": "talker",
    "code_predictor": "code_predictor",
    "audio": "audio",
    "code2wav": "code2wav",
    "visual": "visual",
    "action": "action",
    "mtp_draft": "mtp_draft",
    "dflash_draft": "dflash_draft",
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


def _get_llm_text_config(config: dict) -> dict:
    """Return the promoted text/LLM config dict when present."""
    for key in ("text_config", "llm_config", "language_config"):
        sub = config.get(key)
        if isinstance(sub, dict) and sub.get("hidden_size") is not None:
            return sub
    return config


def _has_mtp(config: dict) -> bool:
    """Return True when the checkpoint exposes the MTP branch."""
    text_cfg = _get_llm_text_config(config)
    return bool(text_cfg.get("mtp_num_hidden_layers") is not None)


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

    if target_text.get("model_type") not in ("gemma4", "gemma4_text"):
        raise ValueError(
            "Gemma4 MTP target must have model_type gemma4/gemma4_text in text_config."
        )
    if assistant_config.get("model_type") != "gemma4_assistant":
        raise ValueError(
            "Gemma4 MTP assistant must have root model_type gemma4_assistant.")
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
    """Load all safetensors shards in *model_dir* into a flat dict."""
    import glob

    from safetensors.torch import load_file

    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shards:
        logger.error("No safetensors files found in %s", model_dir)
        sys.exit(1)
    weights: dict = {}
    for shard in shards:
        logger.info("  Loading shard: %s", os.path.basename(shard))
        weights.update(load_file(shard, device="cpu"))
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


def _patch_multimodal_token_ids(model_dir: str, llm_out_dir: str,
                                model_type: str) -> None:
    """Inject multimodal special-token IDs into the exported LLM's ``config.json``.

    Dispatches to the family-specific collector in priority order:
    thinker_config → Nemotron rename → tokenizer fallback.  Any IDs a
    later collector returns that aren't already set by an earlier one are
    merged in (handles hybrid layouts).
    """
    cfg_path = os.path.join(llm_out_dir, "config.json")
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


def _alpamayo_llm_key_remap(key: str) -> "Optional[str]":
    """Remap ``vlm.lm_head.*`` → ``lm_head.*`` (not covered by prefix detection)."""
    if key.startswith("vlm.lm_head."):
        return key[len("vlm."):]
    return key


def _export_llm(model_dir: str,
                llm_out_dir: str,
                model_type: str = "",
                eagle_base: bool = False,
                fp8_embedding: bool = False,
                reduced_vocab_dir: str = "",
                mtp_base: bool = False,
                dflash_base: bool = False,
                dflash_tree_base: bool = False,
                dflash_draft_dir: str = "",
                gemma4_mtp_base: bool = False,
                externalize_weights: "list[str] | None" = None,
                tp_size: int = 1,
                num_decoder_layers: "int | None" = None) -> None:
    """Export LLM backbone via the standard tensorrt_edgellm pipeline.

    When ``tp_size > 1``, exports ``tp_size`` per-rank ONNX files named
    ``model_tp{N}_rank{R}.onnx`` (matches ``cpp/builder/llmBuilder.cpp``).
    Each rank reloads the checkpoint fresh and shards weights to its
    slice on assignment.
    """
    os.makedirs(llm_out_dir, exist_ok=True)

    key_remap = (_alpamayo_llm_key_remap
                 if model_type == "alpamayo_r1" else None)

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

    # Gemma4 NVFP4 MoE: checkpoint stores router/experts at layer level but
    # model tree nests them under moe_block with _experts indirection.
    # Only activate when the checkpoint is NVFP4-quantized (otherwise the
    # FP16 dense path uses router/experts directly on the layer).
    if key_remap is None and model_type in _GEMMA4_MODEL_TYPES:
        if _is_nvfp4_checkpoint(model_dir):
            config_path = os.path.join(model_dir, "config.json")
            with open(config_path) as f:
                _cfg = json.load(f)
            _llm = _cfg.get("text_config", _cfg)
            # Only MoE checkpoints (e.g. 26B-A4B) nest router/experts under
            # moe_block and need the remap; dense NVFP4 checkpoints (E2B/E4B/
            # 31B) have enable_moe_block=False and export directly.
            if _llm.get("enable_moe_block", False):
                from ..models.gemma4.modeling_gemma4_text import \
                    GEMMA4_NVFP4_KEY_REMAP
                key_remap = GEMMA4_NVFP4_KEY_REMAP

    if tp_size <= 1:
        ranks = [(0, 1)]
        out_paths = [os.path.join(llm_out_dir, "model.onnx")]
    else:
        ranks = [(r, tp_size) for r in range(tp_size)]
        out_paths = [
            os.path.join(llm_out_dir, f"model_tp{tp_size}_rank{r}.onnx")
            for r in range(tp_size)
        ]

    for (rank, world), output_path in zip(ranks, out_paths):
        if world > 1:
            logger.info("[LLM] === rank %d / %d ===", rank, world)

        logger.info("[LLM] Loading checkpoint from %s", model_dir)
        try:
            from ..model import AutoModel
            model = AutoModel.from_pretrained(
                model_dir,
                device="cpu",
                eagle_base=eagle_base,
                key_remap=key_remap,
                reduced_vocab_dir=reduced_vocab_dir or None,
                mtp_base=mtp_base,
                mtp_tree_base=mtp_base,
                dflash_base=dflash_base,
                dflash_tree_base=dflash_tree_base,
                dflash_draft_dir=dflash_draft_dir or None,
                gemma4_mtp_base=gemma4_mtp_base,
                tp_size=world,
                tp_rank=rank,
                num_decoder_layers=num_decoder_layers,
            )
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            logger.exception("[LLM] Failed to load checkpoint")
            raise SystemExit(1) from exc

        # Per-rank runtime config so each rank artifact is self-describing.
        # Single-device exports keep the conventional "config.json".
        config_filename = ("config.json" if world == 1 else
                           f"config_tp{world}_rank{rank}.json")

        logger.info("[LLM] Exporting to %s", output_path)
        try:
            from ..onnx.export import export_onnx
            export_onnx(model,
                        output_path,
                        model_dir=model_dir,
                        fp8_embedding=fp8_embedding,
                        reduced_vocab_dir=reduced_vocab_dir,
                        externalize_weights=externalize_weights,
                        config_filename=config_filename)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.exception("[LLM] ONNX export failed")
            raise SystemExit(1) from exc

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
    _patch_multimodal_token_ids(model_dir, llm_out_dir, model_type)

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
                                          mtp_draft=True,
                                          mtp_tree_base=True)
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
        from safetensors.torch import save_file as _save_safetensors

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
        "gemma4_unified": "gemma4_unified_vision",
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
                      "qwen3_5", "qwen3_5_moe"):
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
        if model_type == "qwen3_omni":
            # Qwen3-Omni video temporal MRoPE scale. HF keeps it at thinker_config
            # level (not vision_config), so copy it into vision_config where the C++
            # Qwen3OmniViTRunner reads it.
            _pips = _thinker_cfg.get("position_id_per_seconds",
                                     config.get("position_id_per_seconds"))
            if _pips is not None:
                vis_cfg_out["vision_config"] = dict(
                    vis_cfg_out["vision_config"])
                vis_cfg_out["vision_config"]["position_id_per_seconds"] = _pips

    if model_type == "gemma4":
        vis_cfg_out["vision_config"] = dict(vis_cfg_out["vision_config"])
        vis_cfg_out["vision_config"]["model_type"] = "gemma4_vision"
        text_cfg = config.get("text_config") or {}
        if text_cfg:
            vis_cfg_out["text_config"] = text_cfg
        if "image_token_id" in config:
            vis_cfg_out["image_token_id"] = config["image_token_id"]
        else:
            image_token_id = _find_token_id(model_dir, "<|image_pad|>")
            if image_token_id is not None:
                vis_cfg_out["image_token_id"] = image_token_id
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
        # additionally reads patch_size and downsample_ratio.
        for key in ("llm_config", "img_context_token_id", "img_start_token_id",
                    "img_end_token_id", "force_image_size", "norm_mean",
                    "norm_std", "patch_size", "downsample_ratio"):
            if key in config:
                vis_cfg_out[key] = config[key]
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


def _extract_tts_weights(model_dir: str, out_dir: str) -> None:
    """Extract TTS-specific weight files from the full checkpoint.

    Saves:
    - ``text_embedding.safetensors``  — thinker text embedding [text_vocab_size, hidden]
    - ``text_projection.safetensors`` — MLP weights (fc1/fc2 weight+bias)
    """
    from safetensors.torch import save_file

    weights = _load_all_weights(model_dir)

    # text_embedding: talker.model.text_embedding.weight [151936, 2048]
    text_emb_key = "talker.model.text_embedding.weight"
    if text_emb_key not in weights:
        logger.error("Key %r not found in checkpoint", text_emb_key)
        sys.exit(1)
    text_emb = weights[text_emb_key].cpu()
    save_file(_to_fp16({"text_embedding": text_emb}),
              os.path.join(out_dir, "text_embedding.safetensors"))
    logger.info("[TTS] Wrote text_embedding.safetensors %s",
                list(text_emb.shape))

    # text_projection: talker.text_projection.linear_fc1/fc2 weight/bias
    proj_keys = {
        "linear_fc1.weight": "talker.text_projection.linear_fc1.weight",
        "linear_fc1.bias": "talker.text_projection.linear_fc1.bias",
        "linear_fc2.weight": "talker.text_projection.linear_fc2.weight",
        "linear_fc2.bias": "talker.text_projection.linear_fc2.bias",
    }
    proj_tensors = {}
    for save_name, ckpt_key in proj_keys.items():
        if ckpt_key not in weights:
            logger.error("Key %r not found in checkpoint", ckpt_key)
            sys.exit(1)
        proj_tensors[save_name] = weights[ckpt_key].cpu()
    save_file(_to_fp16(proj_tensors),
              os.path.join(out_dir, "text_projection.safetensors"))
    logger.info("[TTS] Wrote text_projection.safetensors (4 tensors)")


def _extract_omni_talker_sidecars(model_dir: str, out_dir: str) -> None:
    """Qwen3-Omni Talker weight extractor.

    Qwen3-Omni Talker consumes the thinker's ``hidden_states`` directly
    (instead of a separate text-token embedding lookup like Qwen3-TTS),
    so the sidecars are:

    - ``embedding.safetensors``          — codec token embedding
    - ``hidden_projection.safetensors``  — Omni-only projection from
                                           thinker's hidden_states space
                                           (``2560``) into talker space
                                           (``1024``).
    - ``text_projection.safetensors``    — projection for text tokens
                                           (shared with Qwen3-TTS).
    """
    from safetensors.torch import save_file

    weights = _load_all_weights(model_dir)

    # 1. embedding.safetensors — codec embedding used by the C++ runtime.
    ce_key = "talker.model.codec_embedding.weight"
    if ce_key not in weights:
        logger.error("Key %r not found in checkpoint", ce_key)
        sys.exit(1)
    ce_tensor = weights[ce_key].cpu()
    save_file(_to_fp16({"embedding": ce_tensor}),
              os.path.join(out_dir, "embedding.safetensors"))
    logger.info("[Talker-Omni] Wrote embedding.safetensors %s",
                list(ce_tensor.shape))

    # 2. hidden_projection.safetensors — Omni-only.
    hp_keys = {
        "linear_fc1.weight": "talker.hidden_projection.linear_fc1.weight",
        "linear_fc1.bias": "talker.hidden_projection.linear_fc1.bias",
        "linear_fc2.weight": "talker.hidden_projection.linear_fc2.weight",
        "linear_fc2.bias": "talker.hidden_projection.linear_fc2.bias",
    }
    hp_tensors = {}
    for save_name, ckpt_key in hp_keys.items():
        if ckpt_key not in weights:
            logger.error("Key %r not found in checkpoint", ckpt_key)
            sys.exit(1)
        hp_tensors[save_name] = weights[ckpt_key].cpu()
    save_file(_to_fp16(hp_tensors),
              os.path.join(out_dir, "hidden_projection.safetensors"))
    logger.info(
        "[Talker-Omni] Wrote hidden_projection.safetensors (4 tensors)")

    # 3. text_projection.safetensors — same keys as Qwen3-TTS.
    tp_keys = {
        "linear_fc1.weight": "talker.text_projection.linear_fc1.weight",
        "linear_fc1.bias": "talker.text_projection.linear_fc1.bias",
        "linear_fc2.weight": "talker.text_projection.linear_fc2.weight",
        "linear_fc2.bias": "talker.text_projection.linear_fc2.bias",
    }
    tp_tensors = {}
    for save_name, ckpt_key in tp_keys.items():
        if ckpt_key not in weights:
            logger.error("Key %r not found in checkpoint", ckpt_key)
            sys.exit(1)
        tp_tensors[save_name] = weights[ckpt_key].cpu()
    save_file(_to_fp16(tp_tensors),
              os.path.join(out_dir, "text_projection.safetensors"))
    logger.info("[Talker-Omni] Wrote text_projection.safetensors (4 tensors)")


def _make_talker_sub_config(model_dir: str, sub_path) -> "ModelConfig":
    """Build a :class:`ModelConfig` from a nested sub-config of ``config.json``.

    Only used by multi-stage Talker/CodePredictor exports (Qwen3-Omni), where
    the dense decoder's architecture config lives under ``talker_config.*``
    in the shared root config.  Writes the nested sub-dict into a temp
    ``config.json`` and symlinks the root's safetensors so quant detection
    still works.

    Args:
        model_dir: Directory containing the checkpoint's root ``config.json``.
        sub_path:  Sequence of keys to walk into the root config
                   (e.g. ``["talker_config", "text_config"]``).
    """
    import tempfile

    from ..model import load_model_config

    cfg = _load_config(model_dir)
    for key in sub_path:
        if not isinstance(cfg, dict) or key not in cfg:
            logger.error("sub-config path %s not found in %s/config.json",
                         ".".join(sub_path), model_dir)
            sys.exit(1)
        cfg = cfg[key]

    with tempfile.TemporaryDirectory() as tmp_dir:
        for fname in os.listdir(model_dir):
            if fname.endswith(".safetensors") or fname.endswith(
                    ".safetensors.index.json"):
                src = os.path.join(model_dir, fname)
                dst = os.path.join(tmp_dir, fname)
                if not os.path.exists(dst):
                    os.symlink(src, dst)
        tmp_cfg_path = os.path.join(tmp_dir, "config.json")
        with open(tmp_cfg_path, "w") as f:
            json.dump(cfg, f)
        return load_model_config(tmp_dir)


def _patch_tts_config(model_dir: str, out_dir: str) -> None:
    """Patch the exported config.json with TTS-specific fields.

    Reads the input config and injects codec token IDs, TTS token IDs,
    thinker_hidden_size, and speaker_id mapping into the already-written
    config.json in *out_dir*.

    Accepts two input layouts:
      * HF root config with nested ``talker_config`` (fields under sub-dict).
      * Standalone Talker config from a prior quant export (fields at top
        level, written by ``_write_standalone_talker_config``).
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
    # already had one (e.g. set by ``_write_standalone_talker_config``).
    dsi = pick("default_speaker_id")
    if dsi is not None and "default_speaker_id" not in cfg:
        cfg["default_speaker_id"] = dsi

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
    os.makedirs(llm_out_dir, exist_ok=True)
    output_path = os.path.join(llm_out_dir, "model.onnx")

    logger.info("[Talker] Loading checkpoint from %s (model_type=%s)",
                model_dir, model_type)
    try:
        from ..checkpoint.loader import load_weights
        from ..model import load_model_config
        from ..models.qwen3_tts import TalkerCausalLM

        if model_type in ("qwen3_omni", "qwen3_omni_moe"):
            config = _make_talker_sub_config(model_dir,
                                             ["talker_config", "text_config"])
            extract_sidecars = _extract_omni_talker_sidecars
        else:  # qwen3_tts and any other future dense-talker variants
            config = load_model_config(model_dir)
            extract_sidecars = _extract_tts_weights

        model = TalkerCausalLM(config)
        model.to("cpu")
        load_weights(model,
                     model_dir,
                     device="cpu",
                     key_prefix="talker.",
                     key_remap=_talker_key_remap)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.exception("[Talker] Failed to load checkpoint")
        raise SystemExit(1) from exc

    logger.info("[Talker] Exporting ONNX to %s", output_path)
    try:
        from ..onnx.export import export_onnx
        export_onnx(model, output_path, model_dir=model_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[Talker] ONNX export failed")
        raise SystemExit(1) from exc

    logger.info("[Talker] Extracting weight sidecars ...")
    extract_sidecars(model_dir, llm_out_dir)

    logger.info("[Talker] Patching config.json with TTS fields ...")
    _patch_tts_config(model_dir, llm_out_dir)

    logger.info("[Talker] Done: %s", output_path)


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
    - ``lm_head_weight`` as an ONNX input (dynamic, 15 different heads)
    - ``hidden_states`` as an additional output (for residual connection)
    - MLP FP16 overflow WAR applied to all layers

    Outputs:
    - ``model.onnx`` — CodePredictor ONNX graph
    - ``codec_embeddings.safetensors`` — 15 embedding tables
    - ``lm_heads.safetensors`` — 15 lm_head weights
    - ``small_to_mtp_projection.safetensors`` — talker→CP projection
    - ``config.json`` — LLM config with ``use_embeddings_input: true``
    """
    os.makedirs(cp_out_dir, exist_ok=True)
    output_path = os.path.join(cp_out_dir, "model.onnx")

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

    # Write a temporary config.json for the CodePredictor so ModelConfig can
    # parse it.  The CP sub-config is a valid standalone Qwen3 config.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Copy safetensors index/files for has_qk_norm detection
        for fname in os.listdir(model_dir):
            if fname.endswith(".safetensors") or fname.endswith(
                    ".safetensors.index.json"):
                src = os.path.join(model_dir, fname)
                dst = os.path.join(tmp_dir, fname)
                if not os.path.exists(dst):
                    os.symlink(src, dst)
        # Rewrite ``hf_quant_config.json`` into the CP-relative namespace:
        # strip the CP subtree prefix from exclude entries and drop everything
        # outside CP so ``make_linear`` in the standalone graph resolves.
        hf_quant_src = os.path.join(model_dir, "hf_quant_config.json")
        cp_only_excludes: list = []
        if os.path.exists(hf_quant_src):
            with open(hf_quant_src) as f:
                hf_q = json.load(f)
            q = hf_q.get("quantization", {})
            # Strip either the full-Omni CP prefix or the Talker-only CP
            # prefix; drop entries outside the CP subtree (Talker body,
            # Thinker, code2wav — none exist inside the standalone CP graph).
            cp_prefix = load_key_prefix
            for ex in q.get("exclude_modules", []):
                if ex.startswith(cp_prefix):
                    cp_only_excludes.append(ex[len(cp_prefix):])
            q["exclude_modules"] = cp_only_excludes
            hf_q["quantization"] = q
            with open(os.path.join(tmp_dir, "hf_quant_config.json"), "w") as f:
                json.dump(hf_q, f, indent=2)
        # Write the CP config
        tmp_cfg_path = os.path.join(tmp_dir, "config.json")
        with open(tmp_cfg_path, "w") as f:
            json.dump(cp_cfg, f)

        from ..config import _normalize_module_name
        from ..model import load_model_config
        config = load_model_config(tmp_dir)

    # ``_parse_quant`` auto-detects unquantized weights from the whole
    # checkpoint. Post-normalization those thinker/encoder module names
    # collide with CP paths and force every CP Linear to FP16. Overwrite
    # ``excluded`` with only the CP-scoped entries.
    if config.quant is not None:
        config.quant.excluded = [
            _normalize_module_name(ex) for ex in cp_only_excludes
        ]

    # Override model_type for runtime identification
    config.model_type = _CP_RUNTIME_MODEL_TYPE.get(model_type,
                                                   "qwen3_tts_code_predictor")

    # CP's MLP path is the same for FP16 and FP8: FP32 silu*up + FP32
    # down_proj matmul.  down_proj is always FP16Linear (excluded from FP8
    # quant by ``FP8_CP``), so the FP32 matmul is safe in either mode.
    from ..models.qwen3_tts import CodePredictorCausalLM
    model = CodePredictorCausalLM(config)
    model.to("cpu")

    from ..checkpoint.loader import load_weights
    load_weights(model, model_dir, device="cpu", key_prefix=load_key_prefix)

    logger.info("[CodePredictor] Exporting ONNX to %s", output_path)
    try:
        from ..onnx.export import export_onnx
        export_onnx(model, output_path, model_dir=model_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[CodePredictor] ONNX export failed")
        raise SystemExit(1) from exc

    # ``torch.onnx.export`` drops ``axis=0`` from per-channel DequantizeLinear
    # nodes → TRT engine build fails with ``K == scaleSize``. Restore it.
    _patch_cp_dq_axis(output_path)

    # Extract CodePredictor-specific weight files
    logger.info("[CodePredictor] Extracting weight files ...")
    _extract_code_predictor_weights(model_dir,
                                    cp_out_dir,
                                    talker_cfg,
                                    key_prefix=load_key_prefix)

    # Patch config.json with use_embeddings_input and num_code_groups
    cfg_path = os.path.join(cp_out_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["use_embeddings_input"] = True
        cfg["num_code_groups"] = talker_cfg.get("num_code_groups", 16)
        # CodePredictor has no deepstack visual inputs; override the
        # inherited value from ModelConfig (see note in _patch_tts_config).
        cfg["num_deepstack_features"] = 0
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        logger.info("[CodePredictor] Patched config.json with "
                    "use_embeddings_input and num_code_groups")

    logger.info("[CodePredictor] Done: %s", output_path)


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
    layout) or ``code_predictor.`` (Talker-only submodule layout produced by
    ``qwen3_omni._export_submodel`` for MoE CP-only exports).
    """
    from safetensors.torch import save_file

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
         "code2wav, action. Useful for re-running a single stage, e.g. "
         "``--components code_predictor`` to refresh only the CodePredictor."),
    )
    p.add_argument(
        "--eagle-base",
        action="store_true",
        help=
        "Export as EAGLE3 base model (adds tree-attention I/O and hidden_states output).",
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
         "MTP weights and always use tree verification; paired MTP models such as "
         "Gemma4 require --mtp-draft-dir."),
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
        help=argparse.SUPPRESS,
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
        help=(
            "Tensor-parallel world size (default: 1 = single device). "
            "When >1, exports per-rank LLM ONNX files named "
            "model_tp{N}_rank{R}.onnx (matching cpp/builder/llmBuilder.cpp)."),
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
            "Qwen3.5, Nemotron-H); not for eagle/mtp/dflash "
            "speculative-decoding variants."),
    )
    args = p.parse_args()

    model_dir = _resolve_model_dir(args.model)
    config = _load_config(model_dir)
    model_type: str = config.get("model_type", "unknown")
    dtype = _dtype_from_str(args.dtype)
    has_mtp_draft = _has_mtp(config)
    is_gemma4_target = model_type in _GEMMA4_MODEL_TYPES
    mtp_draft_dir_arg = args.mtp_draft_dir or args.gemma4_mtp_assistant_dir
    gemma4_mtp_requested = args.mtp and is_gemma4_target
    gemma4_mtp_assistant_dir = ""
    gemma4_kv_sharing_map: list[dict] = []
    externalize_weights = resolve_externalize_weights(args.externalize_weights)

    if (model_type == "qwen3_tts"
            and config.get("tts_model_type") != "custom_voice"):
        p.error("Only Qwen3-TTS CustomVoice checkpoints are supported. "
                f"Got tts_model_type={config.get('tts_model_type')!r}.")

    if args.mtp_tree_base:
        args.mtp = True
    if args.eagle_base and args.mtp:
        p.error("--eagle-base and --mtp cannot be enabled together")
    if args.mtp_draft_dir and args.gemma4_mtp_assistant_dir:
        p.error("Use only one MTP draft checkpoint directory option")
    if mtp_draft_dir_arg and args.eagle_base:
        p.error("--mtp-draft-dir cannot be combined with --eagle-base")
    if mtp_draft_dir_arg and (args.dflash_base or args.dflash_draft):
        p.error("--mtp-draft-dir cannot be combined with DFlash export")
    if args.dflash_tree_base:
        args.dflash_base = True
    if args.dflash_base and (args.eagle_base or args.mtp):
        p.error("--dflash-base cannot be combined with --eagle-base or --mtp")
    if args.dflash_draft and (args.eagle_base or args.mtp):
        p.error("--dflash-draft cannot be combined with --eagle-base or --mtp")
    if args.dflash_draft and not args.dflash_draft_dir:
        p.error("--dflash-draft requires --dflash-draft-dir")
    if args.mtp and args.skip_llm:
        p.error("--mtp requires LLM export; remove --skip-llm")
    if mtp_draft_dir_arg and args.skip_llm:
        p.error("--mtp-draft-dir requires LLM export; remove --skip-llm")
    if args.dflash_base and args.skip_llm:
        p.error("--dflash-base requires LLM export; remove --skip-llm")
    if args.dflash_draft and args.skip_llm:
        logger.info(
            "--dflash-draft implies --skip-llm (draft export is independent)")
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
        if args.eagle_base or args.mtp or args.dflash_base or args.dflash_draft:
            p.error("--num-decoder-layer cannot be combined with "
                    "--eagle-base / --mtp / --dflash-base / --dflash-draft")

    _VALID_COMPONENTS = {
        "thinker", "mtp_draft", "talker", "code_predictor", "visual", "audio",
        "code2wav", "action"
    }
    requested_components = {
        c.strip()
        for c in args.components.split(",") if c.strip()
    }
    unknown = requested_components - _VALID_COMPONENTS
    if unknown:
        p.error(f"--components contains unknown values {sorted(unknown)}; "
                f"valid choices: {sorted(_VALID_COMPONENTS)}")

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

    # When --dflash-draft is set, only the dflash_draft stage runs.
    # DFlash draft is a standalone export (like Eagle draft) — no base LLM,
    # visual, audio, or other components needed.
    _draft_only = args.dflash_draft

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
         and not _draft_only and _allow("thinker"), "thinker", lambda out:
         _export_llm(model_dir,
                     out,
                     model_type=model_type,
                     eagle_base=args.eagle_base,
                     mtp_base=args.mtp and not gemma4_mtp_requested,
                     dflash_base=args.dflash_base,
                     dflash_tree_base=args.dflash_tree_base,
                     dflash_draft_dir=args.dflash_draft_dir,
                     gemma4_mtp_base=gemma4_mtp_requested,
                     fp8_embedding=args.fp8_embedding,
                     reduced_vocab_dir=args.reduced_vocab_dir,
                     externalize_weights=externalize_weights,
                     tp_size=args.tp_size,
                     num_decoder_layers=args.num_decoder_layer)),
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
        (_has_llm_component(model_type, "talker") and not args.skip_llm
         and not _draft_only and _allow("talker"), "talker",
         lambda out: _export_talker(model_dir, out, model_type)),
        (_has_llm_component(model_type, "code_predictor") and not args.skip_llm
         and not _draft_only and _allow("code_predictor"), "code_predictor",
         lambda out: _export_code_predictor(model_dir, out, model_type)),
        (_has_visual(model_type) and not args.skip_visual and not _draft_only
         and _allow("visual"), "visual", _export_visual_component),
        (_has_audio(model_type) and not args.skip_audio and not _draft_only
         and _checkpoint_audio_config(config) is not None and _allow("audio"),
         "audio", lambda out: _export_audio(model_dir,
                                            out,
                                            _get_weights(),
                                            config,
                                            model_type,
                                            dtype,
                                            model_config=_get_model_config())),
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
                 "codec_embeddings.safetensors", "lm_heads.safetensors",
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
