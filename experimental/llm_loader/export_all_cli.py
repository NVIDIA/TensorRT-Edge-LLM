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

    # From experimental/ directory inside the repo:
    python -m llm_loader.export_all_cli /path/to/checkpoint /tmp/onnx_out

    # With explicit dtype
    python -m llm_loader.export_all_cli /path/to/Qwen3-VL-7B /tmp/out --dtype float16

    # With reduced vocabulary
    python -m llm_loader.export_all_cli /path/to/model /tmp/out --reduced-vocab-dir /path/to/reduced_vocab

Supported model types
----------------------
VLMs (LLM + visual encoder):
    qwen3_vl, qwen3_omni          (Qwen3-VL / Qwen3-Omni)
    qwen3_5                       (Qwen3.5)
    qwen2_5_vl                    (Qwen2.5-VL)
    internvl_chat                 (InternVL3)
    internvl                      (InternVL3.5)
    phi4mm, phi4_multimodal       (Phi-4 Multimodal)
    NemotronH_Nano_VL_V2, NemotronH_Nano_Omni_Reasoning_V3
                                 (Nemotron-Omni)

Audio models (LLM + audio encoder):
    qwen3_asr, qwen3_omni, qwen3_omni_thinker
    NemotronH_Nano_VL_V2, NemotronH_Nano_Omni_Reasoning_V3
                                 (Nemotron-Omni)

LLM + Talker decoder + Code2Wav (no audio encoder):
    qwen3_tts    (Talker/CodePredictor are LLM decoders; Code2Wav uses speech_tokenizer/)

LLM-only:
    All other model types supported by :mod:`llm_loader.model.AutoModel`.
"""

import argparse
import json
import logging
import os
import sys
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from .config import ModelConfig

from .checkpoint.checkpoint_utils import normalize_rope_scaling_for_runtime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("llm_loader.export_all_cli")

# ---------------------------------------------------------------------------
# Model type classification
# ---------------------------------------------------------------------------

_NEMOTRON_OMNI_MODEL_TYPES = frozenset([
    "NemotronH_Nano_VL_V2",
    "NemotronH_Nano_Omni_Reasoning_V3",
])

_VLM_MODEL_TYPES = frozenset([
    "qwen3_vl",
    "qwen3_omni",
    "qwen3_5",
    "qwen2_5_vl",
    "internvl",
    "internvl_chat",
    "phi4mm",
    "phi4_multimodal",
    "alpamayo_r1",
    *_NEMOTRON_OMNI_MODEL_TYPES,
])

_AUDIO_MODEL_TYPES = frozenset([
    "qwen3_asr",
    "qwen3_omni",
    "qwen3_omni_thinker",
    *_NEMOTRON_OMNI_MODEL_TYPES,
    # qwen3_tts intentionally excluded: Qwen3-TTS has NO audio encoder.
    # Its Talker and CodePredictor are LLM decoders exported via the LLM pipeline.
])

# Subset of audio models whose LLM config follows the Qwen-style
# ``thinker_config`` layout for modality-token IDs and chat-template tokens.
# Excludes Nemotron-Omni, which has its own field names
# (``img_context_token_id`` / ``sound_context_token_id``) at the source-config
# root and is handled by ``_collect_tokens_from_nemotron_root``.
_ASR_LLM_MODEL_TYPES = _AUDIO_MODEL_TYPES - _NEMOTRON_OMNI_MODEL_TYPES

_CODE2WAV_MODEL_TYPES = frozenset([
    "qwen3_omni",
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
}
_DEFAULT_LLM_COMPONENTS = frozenset(["thinker"])


def _has_visual(model_type: str) -> bool:
    return model_type in _VLM_MODEL_TYPES


def _has_audio(model_type: str) -> bool:
    return model_type in _AUDIO_MODEL_TYPES


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
}

# Per-model overrides on top of ``_DEFAULT_LAYOUT``.
# TODO(jiahaoz): remove the ``qwen3_omni`` entry once the legacy
# ``tensorrt_edgellm`` engine-build scripts are retired; they currently
# hardcode these nested paths.
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


def _collect_user_token_id(model_dir: str, root: dict) -> dict:
    """Resolve ``user_token_id`` for Qwen3-ASR / Qwen3-Omni.

    Qwen3-ASR's source HF config does not expose this field at the
    ``thinker_config`` or root level, but the legacy 0.6 reference output
    writes it (default 872 from the ``Qwen3AsrThinkerConfig`` class default
    attribute).  When the config does not carry it, fall back to a BPE
    ``vocab.json`` lookup of the literal ``"user"`` word.
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

    # Qwen3-ASR / Qwen3-Omni byte-parity additions: ``user_token_id`` (with a
    # BPE vocab fallback) plus the ``model: "qwen3asrthinker"`` tag the legacy
    # 0.6 reference carries.  No current C++ reader uses these, but the
    # legacy reference output includes them and downstream tooling matches
    # against that.
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
                mtp_base: bool = False) -> None:
    """Export LLM backbone via the standard llm_loader pipeline."""
    os.makedirs(llm_out_dir, exist_ok=True)
    output_path = os.path.join(llm_out_dir, "model.onnx")

    key_remap = (_alpamayo_llm_key_remap
                 if model_type == "alpamayo_r1" else None)

    logger.info("[LLM] Loading checkpoint from %s", model_dir)
    try:
        from .model import AutoModel
        model = AutoModel.from_pretrained(
            model_dir,
            device="cpu",
            eagle_base=eagle_base,
            key_remap=key_remap,
            reduced_vocab_dir=reduced_vocab_dir or None,
            mtp_base=mtp_base,
        )
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.exception("[LLM] Failed to load checkpoint")
        raise SystemExit(1) from exc

    logger.info("[LLM] Exporting to %s", output_path)
    try:
        from .onnx.export import export_onnx
        export_onnx(model,
                    output_path,
                    model_dir=model_dir,
                    fp8_embedding=fp8_embedding,
                    reduced_vocab_dir=reduced_vocab_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[LLM] ONNX export failed")
        raise SystemExit(1) from exc

    # Patch multimodal token IDs into the LLM config so the C++ runtime
    # can identify which positions in the token stream must be replaced
    # with vision / audio encoder embeddings (see ``llmEngineRunner.cpp``
    # ``audio_token_id`` / ``image_token_id`` lookup).
    #
    # Source of truth: ``thinker_config`` (Qwen3-Omni) or root config
    # (Qwen-VL / Nemotron-Omni).  Naming conventions vary across checkpoints
    # — see :func:`_patch_multimodal_token_ids` for the fallback chain.
    _patch_multimodal_token_ids(model_dir, llm_out_dir, model_type)

    logger.info("[LLM] Done: %s", output_path)


def _export_mtp_draft(model_dir: str, draft_out_dir: str) -> None:
    """Export the MTP draft model."""
    os.makedirs(draft_out_dir, exist_ok=True)
    output_path = os.path.join(draft_out_dir, "model.onnx")

    logger.info("[MTP Draft] Loading checkpoint from %s", model_dir)
    try:
        from .model import AutoModel
        model = AutoModel.from_pretrained(model_dir,
                                          device="cpu",
                                          mtp_draft=True)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        logger.exception("[MTP Draft] Failed to load checkpoint")
        raise SystemExit(1) from exc

    logger.info("[MTP Draft] Exporting to %s", output_path)
    try:
        from .onnx.export import export_onnx
        export_onnx(model, output_path, model_dir=model_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[MTP Draft] ONNX export failed")
        raise SystemExit(1) from exc

    logger.info("[MTP Draft] Done: %s", output_path)


def _export_visual(model_dir: str, visual_out_dir: str, weights: dict,
                   config: dict, model_type: str, dtype: "torch.dtype",
                   model_config: "ModelConfig") -> None:
    """Export visual encoder via from-scratch llm_loader pipeline."""
    os.makedirs(visual_out_dir, exist_ok=True)
    output_path = os.path.join(visual_out_dir, "model.onnx")

    logger.info("[Visual] Exporting %s visual encoder to %s", model_type,
                output_path)
    try:
        from .onnx.export_encoder import export_visual_onnx
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
        "qwen3_omni": "qwen3_omni_vision_encoder",
    }
    top_level_model_type = _VISUAL_MODEL_TYPE_MAP.get(model_type, model_type)
    vis_cfg_out: dict = {
        "model_type": top_level_model_type,
        "vision_config": vis_cfg,
    }
    if model_type in ("qwen2_5_vl", "qwen3_vl", "qwen3_omni", "qwen3_5"):
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
    # Copy preprocessor_config.json to the visual output dir so the C++
    # runtime can find patch_size, image_mean, image_std, etc.  Applies to
    # every visual family (Qwen VL, InternVL, Phi-4mm) — the C++ visual
    # runners all read from this file.
    import shutil
    pp_src = os.path.join(model_dir, "preprocessor_config.json")
    if os.path.exists(pp_src):
        shutil.copy2(pp_src, visual_out_dir)
        logger.info("[Visual] Copied preprocessor_config.json to %s",
                    visual_out_dir)
    else:
        # Newer quantized checkpoints store image processor config inside
        # processor_config.json under the "image_processor" key.  Extract
        # it and write a standalone preprocessor_config.json.
        proc_src = os.path.join(model_dir, "processor_config.json")
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


def _export_audio(model_dir: str, audio_out_dir: str, weights: dict,
                  config: dict, model_type: str, dtype: "torch.dtype") -> None:
    """Export audio encoder via from-scratch llm_loader pipeline."""
    os.makedirs(audio_out_dir, exist_ok=True)
    output_path = os.path.join(audio_out_dir, "model.onnx")

    logger.info("[Audio] Exporting %s audio encoder to %s", model_type,
                output_path)
    try:
        from .onnx.export_encoder import export_audio_onnx
        export_audio_onnx(
            model_dir=model_dir,
            output_path=output_path,
            weights=weights,
            config=config,
            model_type=model_type,
            dtype=dtype,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[Audio] ONNX export failed")
        raise SystemExit(1) from exc
    logger.info("[Audio] Done: %s", output_path)

    # Write config.json for the C++ runtime
    if model_type in _NEMOTRON_OMNI_MODEL_TYPES:
        # Nemotron-Omni carries ``sound_config`` at the root with its own
        # encoder model_type; keep the full root config alongside so the
        # builder sees everything it needs.
        audio_cfg_out = dict(config)
        sound_model_type = config.get("sound_config", {}).get("model_type")
        if sound_model_type is None:
            raise ValueError(
                "sound_config.model_type not found in config.json")
        audio_cfg_out["model_type"] = sound_model_type
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
        # ``user_token_id`` for byte parity with the legacy 0.6 reference.
        audio_cfg_out.update(_collect_user_token_id(model_dir, config))
        # ``text_config.rope_theta`` is read by
        # ``Qwen3OmniAudioRunner::loadConfig`` for MRope initialisation. For
        # Qwen3-ASR/Omni this lives under ``thinker_config.text_config``.
        text_cfg = (thinker_cfg.get("text_config") or config.get("text_config")
                    or {})
        rope_theta = text_cfg.get("rope_theta")
        if rope_theta is not None:
            audio_cfg_out["text_config"] = {"rope_theta": rope_theta}
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
    """Export Code2Wav vocoder via the standalone llm_loader implementation.

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
            from .models.qwen3_tts import export_qwen3_tts_code2wav
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
        from .models.qwen3_omni import build_code2wav, export_code2wav_onnx
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

    # 1. embedding.safetensors — codec embedding (matches the legacy
    # tensorrt_edgellm filename used by the C++ runtime).
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

    from .config import ModelConfig

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
        return ModelConfig.from_pretrained(tmp_dir)


def _patch_tts_config(model_dir: str, out_dir: str) -> None:
    """Patch the exported config.json with TTS-specific fields.

    Reads the root HF config and injects codec token IDs, TTS token IDs,
    thinker_hidden_size, and speaker_id mapping into the already-written
    config.json in *out_dir*.
    """
    root_config = _load_config(model_dir)
    talker_cfg = root_config.get("talker_config", {})

    cfg_path = os.path.join(out_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    # TTS token IDs (from root config)
    for key in ("tts_pad_token_id", "tts_bos_token_id", "tts_eos_token_id"):
        if key in root_config:
            cfg[key] = root_config[key]
    if "tts_model_type" in root_config:
        cfg["tts_model_type"] = root_config["tts_model_type"]

    # Codec token IDs (from talker_config)
    for key in ("codec_nothink_id", "codec_think_bos_id", "codec_think_eos_id",
                "codec_pad_id", "codec_bos_id", "codec_eos_token_id",
                "codec_think_id"):
        if key in talker_cfg:
            cfg[key] = talker_cfg[key]

    # CustomVoice language conditioning map: language name -> codec token id.
    # Required by the runtime to inject the language-row in the 9-row prefix
    # for CustomVoice checkpoints. Optional for non-CustomVoice models.
    if "codec_language_id" in talker_cfg:
        cfg["codec_language_id"] = talker_cfg["codec_language_id"]

    # thinker_hidden_size and text_vocab_size
    # Qwen3-Omni exposes ``thinker_hidden_size`` directly on talker_config;
    # Qwen3-TTS uses ``text_hidden_size`` — accept either.
    if "thinker_hidden_size" in talker_cfg:
        cfg["thinker_hidden_size"] = talker_cfg["thinker_hidden_size"]
    elif "text_hidden_size" in talker_cfg:
        cfg["thinker_hidden_size"] = talker_cfg["text_hidden_size"]

    # ``text_vocab_size`` is the *thinker*'s text vocabulary size (used by
    # the talker to distinguish text tokens from codec tokens).  For
    # Qwen3-TTS this lives at ``talker_config.text_vocab_size``; for
    # Qwen3-Omni it lives at ``thinker_config.text_config.vocab_size``.
    if "text_vocab_size" in talker_cfg:
        cfg["text_vocab_size"] = talker_cfg["text_vocab_size"]
    else:
        thinker_cfg = root_config.get("thinker_config", {}) or {}
        thinker_text = thinker_cfg.get("text_config", {}) or {}
        tv = thinker_text.get("vocab_size") or thinker_cfg.get("vocab_size")
        if tv is not None:
            cfg["text_vocab_size"] = tv

    # Multi-stage talker runtime needs these; both families store them at
    # the talker_config top level.
    for key in ("accept_hidden_layer", "num_code_groups"):
        if key in talker_cfg:
            cfg[key] = talker_cfg[key]

    # The Talker is a text-only decoder with no deepstack visual inputs.
    # Override the (potentially inherited) value from ModelConfig which may
    # mistakenly set 3 due to substring-matching on "qwen3_omni" in the
    # talker sub-config's model_type (``qwen3_omni_talker_text``).
    cfg["num_deepstack_features"] = 0

    # Speaker ID mapping.  Qwen3-TTS stores the name→id dict under ``spk_id``,
    # Qwen3-Omni under ``speaker_id`` (naming convention drift).  Accept
    # either; missing default_speaker_id causes TTS runtime to pick token 0
    # as speaker → Talker generates garbage / fails to emit codec EOS.
    spk_map = talker_cfg.get("spk_id") or talker_cfg.get("speaker_id")
    if isinstance(spk_map, dict) and spk_map:
        cfg["speaker_id"] = spk_map
        if "default_speaker_id" not in cfg:
            cfg["default_speaker_id"] = next(iter(spk_map.values()))

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
        from .checkpoint.loader import load_weights
        from .config import ModelConfig
        from .models.qwen3_tts import TalkerCausalLM

        if model_type == "qwen3_omni":
            config = _make_talker_sub_config(model_dir,
                                             ["talker_config", "text_config"])
            extract_sidecars = _extract_omni_talker_sidecars
        else:  # qwen3_tts and any other future dense-talker variants
            config = ModelConfig.from_pretrained(model_dir)
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
        from .onnx.export import export_onnx
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

    # Build CodePredictor config from the checkpoint's code_predictor sub-config
    root_config = _load_config(model_dir)
    talker_cfg = root_config.get("talker_config", {})
    cp_cfg = talker_cfg.get("code_predictor_config", {})
    if not cp_cfg.get("hidden_size"):
        logger.error("code_predictor_config not found in talker_config")
        sys.exit(1)

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
        # Write the CP config
        tmp_cfg_path = os.path.join(tmp_dir, "config.json")
        with open(tmp_cfg_path, "w") as f:
            json.dump(cp_cfg, f)

        from .config import ModelConfig
        config = ModelConfig.from_pretrained(tmp_dir)

    # Override model_type for runtime identification
    config.model_type = _CP_RUNTIME_MODEL_TYPE.get(model_type,
                                                   "qwen3_tts_code_predictor")

    # Create CodePredictorCausalLM and load weights
    from .models.qwen3_tts import (CodePredictorCausalLM,
                                   apply_code_predictor_mlp_war)

    model = CodePredictorCausalLM(config)
    model.to("cpu")

    from .checkpoint.loader import load_weights
    load_weights(model,
                 model_dir,
                 device="cpu",
                 key_prefix="talker.code_predictor.")

    # Apply MLP FP16 overflow WAR
    apply_code_predictor_mlp_war(model)

    logger.info("[CodePredictor] Exporting ONNX to %s", output_path)
    try:
        from .onnx.export import export_onnx
        export_onnx(model, output_path, model_dir=model_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("[CodePredictor] ONNX export failed")
        raise SystemExit(1) from exc

    # Extract CodePredictor-specific weight files
    logger.info("[CodePredictor] Extracting weight files ...")
    _extract_code_predictor_weights(model_dir, cp_out_dir, talker_cfg)

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


def _extract_code_predictor_weights(model_dir: str, out_dir: str,
                                    talker_cfg: dict) -> None:
    """Extract codec_embeddings, lm_heads, and small_to_mtp_projection."""
    from safetensors.torch import save_file

    weights = _load_all_weights(model_dir)

    # codec_embeddings: talker.code_predictor.model.codec_embedding.{i}.weight
    num_code_groups = talker_cfg.get("num_code_groups", 16)
    num_embeddings = num_code_groups - 1  # 15 for TTS (16-1=15)
    embedding_dict = {}
    for i in range(num_embeddings):
        key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
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

    # lm_heads: talker.code_predictor.lm_head.{i}.weight
    lm_head_dict = {}
    for i in range(num_embeddings):
        key = f"talker.code_predictor.lm_head.{i}.weight"
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

    # small_to_mtp_projection: talker.code_predictor.small_to_mtp_projection
    proj_w_key = "talker.code_predictor.small_to_mtp_projection.weight"
    proj_b_key = "talker.code_predictor.small_to_mtp_projection.bias"
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


def _build_action_config(root_cfg: dict, weights: dict) -> "ActionConfig":
    """Build an ActionConfig from the Alpamayo root config and weight dict."""
    from .config import ActionConfig

    expert_cfg = root_cfg.get("expert_cfg", {})

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
            num_kv_heads = v.shape[0] // expert_cfg.get("head_dim", 128)
            break

    traj_token_start_idx = root_cfg.get("traj_token_start_idx", 0)
    traj_cfg = root_cfg.get("traj_tokenizer_cfg", {})
    num_bins = traj_cfg.get("num_bins", 0)
    traj_token_start = traj_token_start_idx + num_bins

    in_proj_cfg = root_cfg.get("action_in_proj_cfg", {})

    return ActionConfig(
        rope_theta=5_000_000.0,
        mrope_section=[24, 20, 20],
        mrope_interleaved=True,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=expert_cfg.get("num_attention_heads", 0),
        num_key_value_heads=num_kv_heads,
        head_dim=expert_cfg.get("head_dim", 128),
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
        from .onnx.export_encoder import (export_action_onnx,
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
    p = argparse.ArgumentParser(
        prog="python -m llm_loader.export_all_cli",
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
        "--mtp",
        action="store_true",
        help=
        ("Export MTP components from a single checkpoint (llm/ as mtp_base + mtp_draft/)."
         ),
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
    args = p.parse_args()

    model_dir = _resolve_model_dir(args.model)
    config = _load_config(model_dir)
    model_type: str = config.get("model_type", "unknown")
    dtype = _dtype_from_str(args.dtype)
    has_mtp_draft = _has_mtp(config)

    if (model_type == "qwen3_tts"
            and config.get("tts_model_type") != "custom_voice"):
        p.error("Only Qwen3-TTS CustomVoice checkpoints are supported. "
                f"Got tts_model_type={config.get('tts_model_type')!r}.")

    if args.eagle_base and args.mtp:
        p.error("--eagle-base and --mtp cannot be enabled together")
    if args.mtp and args.skip_llm:
        p.error("--mtp requires LLM export; remove --skip-llm")
    if args.mtp and not has_mtp_draft:
        p.error("--mtp was requested, but the checkpoint does not expose "
                "MTP weights/config")

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
            from .config import ModelConfig
            _model_config[0] = ModelConfig.from_pretrained(model_dir)
        return _model_config[0]

    def _get_code2wav_weights() -> dict:
        if model_type == "qwen3_tts":
            return {}
        return _get_weights()

    # Each stage is (enabled, component_name, exporter_callable).  Exporter
    # receives the computed output dir; the (enabled, component) columns also
    # drive both the pre-run log and the post-run summary below.
    stages = [
        (_has_llm_component(model_type, "thinker")
         and not args.skip_llm, "thinker",
         lambda out: _export_llm(model_dir,
                                 out,
                                 model_type=model_type,
                                 eagle_base=args.eagle_base,
                                 mtp_base=args.mtp,
                                 fp8_embedding=args.fp8_embedding,
                                 reduced_vocab_dir=args.reduced_vocab_dir)),
        (args.mtp, "mtp_draft", lambda out: _export_mtp_draft(model_dir, out)),
        (_has_llm_component(model_type, "talker") and not args.skip_llm,
         "talker", lambda out: _export_talker(model_dir, out, model_type)),
        (_has_llm_component(model_type, "code_predictor")
         and not args.skip_llm, "code_predictor",
         lambda out: _export_code_predictor(model_dir, out, model_type)),
        (_has_visual(model_type) and not _is_alpamayo(model_type)
         and not args.skip_visual, "visual",
         lambda out: _export_visual(model_dir,
                                    out,
                                    _get_weights(),
                                    config,
                                    model_type,
                                    dtype,
                                    model_config=_get_model_config())),
        (_has_visual(model_type) and _is_alpamayo(model_type)
         and not args.skip_visual, "visual",
         lambda out: _export_alpamayo_visual(model_dir,
                                             out,
                                             _get_weights(),
                                             config,
                                             dtype,
                                             model_config=_get_model_config())
         ),
        (_has_audio(model_type)
         and not args.skip_audio, "audio", lambda out: _export_audio(
             model_dir, out, _get_weights(), config, model_type, dtype)),
        (_has_code2wav(model_type) and not args.skip_code2wav, "code2wav",
         lambda out: _export_code2wav(model_dir, out, _get_code2wav_weights(),
                                      config, model_type, dtype)),
        (_has_action(model_type) and not args.skip_action, "action", lambda
         out: _export_action(model_dir,
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
    logger.info("MTP export    : %s", "yes" if args.mtp else "no")
    logger.info("Reduced vocab : %s",
                args.reduced_vocab_dir if args.reduced_vocab_dir else "no")
    logger.info("=" * 60)

    # ``--fp8-embedding`` only applies to the LLM thinker.  Models without a
    # thinker (e.g. Qwen3-TTS) silently fall back to FP16 — warn so the user
    # isn't surprised when the flag has no effect.
    if args.fp8_embedding and not _has_llm_component(model_type, "thinker"):
        logger.warning(
            "--fp8-embedding is not supported for Talker / CodePredictor; "
            "using FP16 embeddings.")

    for enabled, component, fn in stages:
        if enabled:
            fn(
                os.path.join(args.output_dir,
                             _layout_for(model_type, component)))

    # Summary
    _SIDECARS = ("embedding.safetensors", "text_embedding.safetensors",
                 "text_projection.safetensors",
                 "hidden_projection.safetensors",
                 "codec_embeddings.safetensors", "lm_heads.safetensors",
                 "small_to_mtp_projection.safetensors")
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
