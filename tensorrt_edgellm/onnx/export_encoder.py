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
ONNX export for from-scratch visual and audio encoders.

Replaces the former ``export_visual.py`` / ``export_audio.py`` pair with a
single unified module.  Both encoder types share the same dynamo export path;
the only difference is how the model is built and how its I/O spec is provided.

Visual encoders — I/O spec via ``model.get_onnx_export_args(config, device)``:
    - Qwen3-VL         (model_type ``qwen3_vl``, ``qwen3_omni``)
    - Qwen3.5          (model_type ``qwen3_5``, ``qwen3_5_moe``)
    - Qwen2.5-VL       (model_type ``qwen2_5_vl``)
    - InternVL3        (model_type ``internvl_chat``)
    - InternVL3 HF     (model_type ``internvl``)
    - Phi-4 Multimodal (model_type ``phi4mm``, ``phi4_multimodal``)
    - Gemma4            (model_type ``gemma4``)
    - Gemma4 Unified    (model_type ``gemma4_unified``)
    - Nemotron-Omni    (model_type ``NemotronH_Nano_VL_V2`` or
      ``NemotronH_Nano_Omni_Reasoning_V3``)

Audio encoders — I/O spec defined internally or via ``model.get_onnx_export_args``:
    - Qwen3-ASR    (model_type ``qwen3_asr``)
    - Qwen3-Omni   (model_type ``qwen3_omni``, ``qwen3_omni_thinker``)
    - Gemma4 Unified (model_type ``gemma4_unified``)
    - Nemotron-Omni (model_type ``NemotronH_Nano_VL_V2`` or
      ``NemotronH_Nano_Omni_Reasoning_V3``)

Note: Qwen3-TTS has NO audio encoder.  Its Talker/CodePredictor are LLM
decoders exported via the standard LLM pipeline.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .dynamo_translations import build_custom_translation_table
from .export import (_OPSET_VERSION, _fix_initializer_dtypes,
                     _fix_nvfp4_weight_dtype, _permissive_inline_opset,
                     _strip_onnxscript_internal_attrs,
                     setup_fp8_qkv_scales_for_export)

if TYPE_CHECKING:
    from ..config import ModelConfig

logger = logging.getLogger(__name__)

__all__ = [
    "export_visual_onnx",
    "export_audio_onnx",
    "export_action_onnx",
]

# ---------------------------------------------------------------------------
# Visual encoder registry
# ---------------------------------------------------------------------------

_NEMOTRON_OMNI_MODEL_TYPES: frozenset[str] = frozenset([
    "NemotronH_Nano_VL_V2",
    "NemotronH_Nano_Omni_Reasoning_V3",
])

# Maps model_type → internal family name
_VISUAL_REGISTRY: dict[str, str] = {
    "qwen3_vl": "qwen3_vl",
    # Qwen3-Omni (dense and MoE) reuses Qwen3-VL's computation graph but
    # uses HF Qwen3-Omni parameter naming (``thinker.visual.merger.ln_q``,
    # ``mlp.0/2``, ``merger_list``).  Dispatch to a dedicated family that
    # translates ckpt keys before delegating to build_qwen3_vl_visual.
    "qwen3_omni": "qwen3_omni",
    "qwen3_omni_moe": "qwen3_omni",
    "qwen3_omni_next": "qwen3_omni_next",
    "qwen3_5": "qwen3_5",
    "qwen3_5_moe": "qwen3_5",
    "qwen2_5_vl": "qwen2_5_vl",
    "internvl_chat": "internvl3",
    "internvl": "internvl3_5",
    "phi4mm": "phi4mm",
    "phi4_multimodal": "phi4mm",
    "gemma4": "gemma4",
    "gemma4_unified": "gemma4_unified",
    "NemotronH_Nano_VL_V2": "nemotron_omni",
    "NemotronH_Nano_Omni_Reasoning_V3": "nemotron_omni",
    # Cosmos3-Edge reasoner: SigLIP2 packed-patch ViT + Qwen3-VL-style
    # PatchMerger, exported on the qwen3_vl visual ONNX I/O contract.
    "cosmos3_edge": "cosmos3_reasoner",
}

# Maps family → dotted module path inside tensorrt_edgellm
_VISUAL_FAMILY_MODULE: dict[str, str] = {
    "qwen3_vl":
    "tensorrt_edgellm.models.qwen3_vl.modeling_qwen3_vl_visual",
    "qwen3_omni":
    "tensorrt_edgellm.models.qwen3_omni.modeling_qwen3_omni_visual",
    "qwen3_omni_next":
    "tensorrt_edgellm.models.qwen3_omni_next.modeling_qwen3_omni_next_visual",
    "qwen3_5":
    "tensorrt_edgellm.models.qwen3_5.modeling_qwen3_5_visual",
    "qwen2_5_vl":
    "tensorrt_edgellm.models.qwen2_5_vl.modeling_qwen2_5_vl_visual",
    "internvl3":
    "tensorrt_edgellm.models.internvl3.modeling_internvl3_visual",
    "internvl3_5":
    "tensorrt_edgellm.models.internvl3_5.modeling_internvl3_5_visual",
    "phi4mm":
    "tensorrt_edgellm.models.phi4mm.modeling_phi4mm_visual",
    "gemma4":
    "tensorrt_edgellm.models.gemma4.modeling_gemma4_visual",
    "gemma4_unified":
    "tensorrt_edgellm.models.gemma4.modeling_gemma4_unified_visual",
    "nemotron_omni":
    "tensorrt_edgellm.models.nemotron_omni.modeling_nemotron_omni_visual",
    "cosmos3_reasoner":
    "tensorrt_edgellm.models.cosmos3_reasoner.modeling_cosmos3_reasoner_visual",
}

# Maps family → build function name in that module
_VISUAL_FAMILY_BUILD_FN: dict[str, str] = {
    "qwen3_vl": "build_qwen3_vl_visual",
    "qwen3_omni": "build_qwen3_omni_visual",
    "qwen3_omni_next": "build_qwen3_omni_next_visual",
    "qwen3_5": "build_qwen3_5_visual",
    "qwen2_5_vl": "build_qwen25_vl_visual",
    "internvl3": "build_internvl_visual",
    "internvl3_5": "build_internvl3_5_visual",
    "phi4mm": "build_phi4mm_visual",
    "gemma4": "build_gemma4_visual",
    "gemma4_unified": "build_gemma4_unified_visual",
    "nemotron_omni": "build_nemotron_omni_visual",
    "cosmos3_reasoner": "build_cosmos3_reasoner_visual",
}

# ---------------------------------------------------------------------------
# Audio encoder registry
# ---------------------------------------------------------------------------

_AUDIO_MODEL_TYPES: frozenset[str] = frozenset([
    "nemotron3_5_asr",
    "qwen3_asr",
    "qwen3_omni",
    "qwen3_omni_thinker",
    "qwen3_omni_moe",
    "qwen3_omni_moe_thinker",
    "gemma4",
    "qwen3_omni_next",
    "qwen3_omni_next_thinker",
    "gemma4_unified",
    *_NEMOTRON_OMNI_MODEL_TYPES,
    # qwen3_tts intentionally excluded: Qwen3-TTS has NO audio encoder.
])

# Safetensors key prefix per audio model type.
# Qwen3-ASR checkpoints (including 0.6B) embed the audio tower under
# ``thinker.audio_tower.*``, matching the Qwen3-Omni layout.
_AUDIO_KEY_PREFIX: dict[str, str] = {
    "qwen3_asr": "thinker.audio_tower.",
    "qwen3_omni": "thinker.audio_tower.",
    "qwen3_omni_thinker": "thinker.audio_tower.",
    "qwen3_omni_moe": "thinker.audio_tower.",
    "qwen3_omni_moe_thinker": "thinker.audio_tower.",
    "gemma4": "model.audio_tower.",
    "qwen3_omni_next": "thinker.audio_tower.",
    "qwen3_omni_next_thinker": "thinker.audio_tower.",
}

# ---------------------------------------------------------------------------
# Visual config extraction
# ---------------------------------------------------------------------------


def _get_visual_config(model_type: str, config: dict) -> dict:
    """Extract visual encoder sub-config from the full model config."""
    if model_type in ("qwen3_vl", "qwen3_omni", "qwen3_omni_moe",
                      "qwen3_omni_next", "qwen3_5", "qwen3_5_moe",
                      "qwen2_5_vl"):
        # Qwen3-Omni (dense + MoE) / Qwen3-Next Omni store vision_config nested under
        # thinker_config; other Qwen VL variants keep it at the root.
        return (config.get("vision_config")
                or config.get("thinker_config", {}).get("vision_config")
                or config)
    if (model_type in ("internvl", "internvl_chat", "gemma4", "gemma4_unified",
                       "cosmos3_edge")
            or model_type in _NEMOTRON_OMNI_MODEL_TYPES):
        # InternVL / Gemma4 / Nemotron-Omni / Cosmos3-Edge need the full
        # config (vision + text + projection/runtime fields).
        return config
    if model_type in ("phi4mm", "phi4_multimodal"):
        # Phi-4mm visual config is hardcoded (not in config.json).
        # Values match vision_siglip_navit.py::get_siglip_vision_model().
        vcfg: dict = {
            "hidden_size": 1152,
            "image_size": 448,
            "intermediate_size": 4304,
            "num_attention_heads": 16,
            "num_hidden_layers": 27,
            "patch_size": 14,
            "feature_layer": -2,
        }
        vcfg["proj_hidden_size"] = config["hidden_size"]
        return vcfg
    return config


# ---------------------------------------------------------------------------
# Shared dynamo export helper
# ---------------------------------------------------------------------------


def _run_dynamo_export(
    model: nn.Module,
    dynamo_inputs: tuple | dict[str, torch.Tensor],
    output_path: str,
    onnx_input_names: list[str],
    output_names: list[str],
    dynamic_shapes: dict,
) -> None:
    model.eval()
    translation_table = build_custom_translation_table()

    def _count_export_leaves(value, *, root: bool = False) -> int:
        if root and isinstance(value, dict):
            return len(value)
        if isinstance(value, (tuple, list)):
            return sum(_count_export_leaves(v) for v in value)
        return 1

    num_dynamo_inputs = _count_export_leaves(dynamo_inputs, root=True)
    num_dynamic_shapes = _count_export_leaves(dynamic_shapes, root=True)
    assert num_dynamo_inputs == len(onnx_input_names) and \
        num_dynamo_inputs == num_dynamic_shapes, \
        f"dynamo_inputs: {num_dynamo_inputs}, onnx_input_names: {len(onnx_input_names)}, dynamic_shapes: {num_dynamic_shapes}"

    if isinstance(dynamo_inputs, dict):
        args = ()
        kwargs = dynamo_inputs
    else:
        assert isinstance(dynamo_inputs, tuple)
        args = dynamo_inputs
        kwargs = {}

    logger.info("Exporting ONNX to %s (opset %d) ...", output_path,
                _OPSET_VERSION)
    with _permissive_inline_opset():
        prog = torch.onnx.export(
            model,
            args,
            kwargs=kwargs,
            dynamo=True,
            input_names=onnx_input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            opset_version=_OPSET_VERSION,
            custom_translation_table=translation_table,
            external_data=True,
            optimize=True,
        )
    prog.save(output_path, external_data=True)
    with open(output_path, "rb") as _f:
        os.fsync(_f.fileno())
    logger.info("Export complete: %s", output_path)


# ---------------------------------------------------------------------------
# Visual encoder export
# ---------------------------------------------------------------------------


def export_visual_onnx(
    model_dir: str,
    output_path: str,
    weights: dict,
    config: dict,
    model_type: str,
    model_config: "ModelConfig",
    dtype: torch.dtype = torch.float16,
) -> None:
    """Export a from-scratch visual encoder to ONNX.

    Args:
        model_dir:    Source checkpoint directory (for reference; weights
                      already loaded by the caller).
        output_path:  Destination ``.onnx`` file path.
        weights:      Flat ``{key: tensor}`` dict loaded from safetensors.
        config:       Full model ``config.json`` dict.
        model_type:   Value of ``config.json["model_type"]``.
        model_config: Top-level ``ModelConfig``.  All family ``build_fn``s
                      accept it and dispatch their linear layers through
                      ``make_linear``; an FP16 checkpoint produces
                      ``FP16Linear`` everywhere.
        dtype:        Weight dtype (default ``float16``).
    """
    device = "cpu"
    if model_type not in _VISUAL_REGISTRY:
        raise ValueError(f"Unsupported visual model_type {model_type!r}. "
                         f"Supported: {sorted(_VISUAL_REGISTRY)}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    family = _VISUAL_REGISTRY[model_type]
    vcfg = _get_visual_config(model_type, config)

    # Build the from-scratch model using a package-relative import so source
    # checkouts and installed wheels resolve the same implementation.
    _PKG_ROOT = "tensorrt_edgellm"
    _rel = "." + _VISUAL_FAMILY_MODULE[family][len(_PKG_ROOT):]
    mod = importlib.import_module(_rel, package=__package__)
    build_fn = getattr(mod, _VISUAL_FAMILY_BUILD_FN[family])
    logger.info("Building %s visual model ...", family)
    visual_model: nn.Module = build_fn(vcfg,
                                       weights,
                                       model_config=model_config,
                                       dtype=dtype)
    visual_model = visual_model.to(device)
    visual_model.eval()

    # Surface FP8 MHA scales onto attention modules so dynamo sees them as
    # Python-float constants. No-op when FP8 MHA is not enabled.
    setup_fp8_qkv_scales_for_export(visual_model)

    # I/O spec is provided by the model class
    dynamo_inputs, onnx_input_names, output_names, dynamic_shapes = (
        visual_model.get_onnx_export_args(vcfg, device))

    _run_dynamo_export(visual_model, dynamo_inputs, output_path,
                       onnx_input_names, output_names, dynamic_shapes)

    # TRT-compat post-processing for quantized weights — same passes the LLM
    # path runs in ``export.py``.  Without these, NVFP4 weights stay as INT8
    # initializers and TRT's DequantizeLinear can't expand the packed dim.
    if model_config is not None:
        nvfp4 = model_config.quant.uses_nvfp4_weights
        mxfp8 = model_config.quant.uses_mxfp8_weights
        if nvfp4:
            _fix_nvfp4_weight_dtype(output_path)
        if mxfp8:
            _strip_onnxscript_internal_attrs(output_path)
        if nvfp4 or mxfp8:
            # Visual graphs legitimately keep FP32 constants from in-body
            # ``.float()`` casts (RMSNorm computes in FP32); skip the FP32→FP16
            # downgrade that the LLM path uses for tied-lm_head BF16 fixup.
            _fix_initializer_dtypes(output_path,
                                    dedup_dql_scales=True,
                                    cast_fp32_weights_to_fp16=False)

    # Phi-4mm sidecar tensors (GN projection weights)
    if family == "phi4mm":
        sidecar_path = visual_model.save_onnx_sidecar(
            os.path.dirname(os.path.abspath(output_path)))
        logger.info("Saved Phi-4mm GN projection sidecar: %s", sidecar_path)

    # Nemotron-Omni sidecar tensors (runtime-executed patch embedder GEMM
    # weights + raw pos_embed; see modeling_nemotron_omni_visual docstring).
    if family == "nemotron_omni":
        sidecar_path = visual_model.save_onnx_sidecar(
            os.path.dirname(os.path.abspath(output_path)))
        logger.info("Saved Nemotron-Omni embedder sidecar: %s", sidecar_path)


# ---------------------------------------------------------------------------
# Audio encoder export
# ---------------------------------------------------------------------------


def export_audio_onnx(
    model_dir: str,
    output_path: str,
    weights: dict,
    config: dict,
    model_type: str,
    model_config: "ModelConfig | None" = None,
    dtype: torch.dtype = torch.float16,
) -> None:
    """Export a from-scratch audio encoder to ONNX.

    Args:
        model_dir:   Source checkpoint directory (for reference).
        output_path: Destination ``.onnx`` file path.
        weights:     Flat ``{key: tensor}`` dict from safetensors.
        config:      Full ``config.json`` dict.
        model_type:  Value of ``config.json["model_type"]``.
        model_config: Top-level :class:`ModelConfig` for ``make_linear``
            dispatch in :class:`QwenAudioEncoder` Linears (``None`` =
            FP16-only).
        dtype:       Weight dtype (default ``float16``).
    """
    device = "cpu"
    if model_type not in _AUDIO_MODEL_TYPES:
        raise ValueError(f"Unsupported audio model_type {model_type!r}. "
                         f"Supported: {sorted(_AUDIO_MODEL_TYPES)}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Gemma4 Unified is encoder-free: it builds its own module and export
    # arguments, so it bypasses the build_fn registry below entirely.
    if model_type == "gemma4_unified":
        from ..models.gemma4.modeling_gemma4_unified_audio import \
            build_gemma4_unified_audio
        if model_config is None:
            from ..model import load_model_config
            model_config = load_model_config(model_dir)
        logger.info("Building Gemma4 Unified encoder-free audio model ...")
        audio_model = build_gemma4_unified_audio(config,
                                                 weights,
                                                 model_config=model_config,
                                                 dtype=dtype)
        audio_model = audio_model.to(device).eval()
        args, input_names, output_names, dynamic_shapes = (
            audio_model.get_onnx_export_args(config, device))
        _run_dynamo_export(audio_model, args, output_path, input_names,
                           output_names, dynamic_shapes)
        return

    build_fn = None
    extra_kwargs = {}
    if model_type == "nemotron3_5_asr":
        # FastConformer encoder for the RNN-T ASR model. Output frames feed
        # the RNN-T joint network (see modeling_nemotron3_5_asr_decoder.py),
        # not LLM prompt embeddings.
        from ..models.nemotron3_5_asr.modeling_nemotron3_5_asr_audio import \
            build_nemotron3_5_asr_audio
        logger.info("Building Nemotron-3.5-ASR FastConformer encoder ...")
        build_fn = build_nemotron3_5_asr_audio
    elif model_type in _NEMOTRON_OMNI_MODEL_TYPES:
        from ..models.nemotron_omni.modeling_nemotron_omni_audio import \
            build_nemotron_omni_audio
        logger.info("Building Nemotron-Omni audio encoder ...")
        build_fn = build_nemotron_omni_audio
    elif model_type == "gemma4":
        from ..models.gemma4.modeling_gemma4_audio import build_gemma4_audio
        if model_config is None:
            from ..model import load_model_config
            model_config = load_model_config(model_dir)
        key_prefix = _AUDIO_KEY_PREFIX.get(model_type)
        logger.info("Building Gemma4 audio encoder (prefix=%r) ...",
                    key_prefix)
        build_fn = build_gemma4_audio
        extra_kwargs = {
            "prefix": key_prefix,
            "model_config": model_config,
        }
    elif model_type in ("qwen3_omni_next", "qwen3_omni_next_thinker"):
        # Qwen3-Next audio encoder adds a 4th stride-2 Conv2d (16x downsample
        # instead of Qwen3-ASR/Qwen3-Omni's 8x); the builder + subclass
        # ``get_onnx_export_args`` override handle the shape difference.
        from ..models.qwen3_omni_next.modeling_qwen3_omni_next_audio import \
            build_qwen3_omni_next_audio
        config = config.get("thinker_config",
                            {}).get("audio_config",
                                    config.get("audio_config", config))
        key_prefix = _AUDIO_KEY_PREFIX.get(model_type)
        logger.info("Building %s audio encoder (prefix=%r) ...", model_type,
                    key_prefix)
        build_fn = build_qwen3_omni_next_audio
        extra_kwargs = {
            "prefix": key_prefix,
            "model_config": model_config,
        }
    else:
        from ..models.qwen3_asr.modeling_qwen3_asr_audio import \
            build_qwen_audio
        config = config.get("thinker_config",
                            {}).get("audio_config",
                                    config.get("audio_config", config))
        key_prefix = _AUDIO_KEY_PREFIX.get(model_type)
        logger.info("Building %s audio encoder (prefix=%r) ...", model_type,
                    key_prefix)
        build_fn = build_qwen_audio
        extra_kwargs = {
            "prefix": key_prefix,
            "model_config": model_config,
        }

    audio_model = build_fn(config, weights, dtype, **extra_kwargs)
    audio_model = audio_model.to(device).eval()
    args, input_names, output_names, dynamic_shapes = (
        audio_model.get_onnx_export_args(config, device))

    _run_dynamo_export(audio_model, args, output_path, input_names,
                       output_names, dynamic_shapes)


# ---------------------------------------------------------------------------
# Action expert export (Alpamayo)
# ---------------------------------------------------------------------------


def export_action_onnx(
    output_path: str,
    weights: dict,
    config: "ActionConfig",
    max_kv_cache_capacity: int,
    dtype: torch.dtype = torch.float16,
) -> None:
    """Export Alpamayo action expert (one flow-matching step) to ONNX.

    Args:
        output_path: Destination ``.onnx`` file path.
        weights:     Flat ``{key: tensor}`` dict from safetensors (must include
                     ``expert.*``, ``action_in_proj.*``, ``action_out_proj.*``).
        config:      :class:`~config.ActionConfig` with expert hyperparameters.
        max_kv_cache_capacity: Fixed KV cache capacity (must match LLM engine).
        dtype:       Weight dtype (default ``float16``).
    """
    device = "cpu"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    from ..models.alpamayo.modeling_alpamayo_action import \
        build_alpamayo_1_action
    logger.info("Building Alpamayo action expert ...")
    model = build_alpamayo_1_action(config, weights, dtype)
    model = model.to(device)
    model.eval()

    dynamo_inputs, onnx_input_names, output_names, dynamic_shapes = (
        model.get_onnx_export_args(max_kv_cache_capacity, device))

    _run_dynamo_export(model, dynamo_inputs, output_path, onnx_input_names,
                       output_names, dynamic_shapes)


def write_action_config(config: "ActionConfig", max_kv_cache_capacity: int,
                        out_dir: str) -> None:
    """Write the action config.json for the C++ runtime."""
    import json
    os.makedirs(out_dir, exist_ok=True)
    rope_scaling = {
        "mrope_section": config.mrope_section,
        "mrope_interleaved": config.mrope_interleaved,
        "rope_type": "mrope",
        "type": "mrope",
    }
    cfg_out = {
        "rope_theta": config.rope_theta,
        "rope_scaling": rope_scaling,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "rms_norm_eps": config.rms_norm_eps,
        "num_traj_tokens": config.num_traj_tokens,
        "traj_token_start": config.traj_token_start,
        "n_diffusion_tokens": config.n_diffusion_tokens,
        "builder_config": {
            "max_kv_cache_capacity": max_kv_cache_capacity,
        },
    }
    path = os.path.join(out_dir, "config.json")
    with open(path, "w") as f:
        json.dump(cfg_out, f, indent=2)
    logger.info("Wrote action config.json: %s", path)
