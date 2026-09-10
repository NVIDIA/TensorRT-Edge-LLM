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
"""Quantization recipe configurations for ModelOpt.

``quant_cfg`` is an ordered list of rule entries
(``{"quantizer_name": <glob>, "cfg": <attrs>?, "enable": <bool>?}``) applied
last-match-wins. We start from a stock named config (``mtq.FP8_DEFAULT_CFG`` /
``NVFP4_DEFAULT_CFG`` / ...) and append the overrides Edge-LLM needs, deriving
per-submodule precision from that same config so bit-widths are never restated.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import modelopt.torch.quantization as mtq

QuantCfgEntry = Dict[str, Any]

# Backbone recipes. Deep-copied per call so appended overrides never mutate the
# shared module-level singletons.
_BACKBONE_CFG_MAP = {
    "fp8": mtq.FP8_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    "mxfp8": mtq.MXFP8_DEFAULT_CFG,
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
}

# Methods exposed on each per-submodule override surface.
_LM_HEAD_METHODS = ("fp8", "int4_awq", "nvfp4", "mxfp8")
_VISUAL_METHODS = ("fp8", )
_AUDIO_METHODS = ("fp8", )
_CP_METHODS = ("fp8", )

# Visual submodule prefixes across HuggingFace VLM families (Qwen-VL ``visual``,
# InternVL ``vision_tower`` / ``multi_modal_projector`` / ``vision_model`` /
# ``mlp1``, Phi-4mm ``image_embed``, Gemma4 ``embed_vision``).
_VISUAL_PREFIXES = (
    "visual",
    "vision_tower",
    "vision_model",
    "multi_modal_projector",
    "mlp1",
    "image_embed",
    "embed_vision",
)
_VISUAL_PATTERNS = tuple(f"*{p}.*" for p in _VISUAL_PREFIXES)

# Audio submodule prefixes (Qwen3-ASR/Omni audio encoder, Phi-4mm audio_embed,
# Gemma4 embed_audio).
_AUDIO_PREFIXES = (
    "audio_tower",
    "audio_embed",
    "embed_audio",
)
_AUDIO_PATTERNS = tuple(f"*{p}.*" for p in _AUDIO_PREFIXES)

# Qwen3-Omni Talker CodePredictor.
_CP_PREFIXES = ("code_predictor", )
_CP_PATTERNS = tuple(f"*{p}.*" for p in _CP_PREFIXES)

# Linear submodules where FP8 loses precision: down_proj (silu*up ∈
# [-39, 72] → per-tensor amax quantizes to garbage) and lm_head[0..14]
# (each codebook sees 1/15 of calib signal, amax undertrained).
# talker_projection runs as an fp16 sidecar GEMM in the C++ runtime, so
# quantizing it only adds error without any kernel to use the FP8 weights.
_CP_LINEAR_EXCLUDES = ("lm_head", "down_proj", "talker_projection")

# CP attention BMM quantizers (mixed-precision KV rejected by ONNX export).
_CP_BMM_EXCLUDES = ("q_bmm", "k_bmm", "v_bmm")

# Qwen3-Omni Code2Wav codec->waveform decoder. Always disabled today.
_CODE2WAV_PATTERNS = ("*code2wav.*", )


def _method_weight_input_cfgs(
        method: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """``(weight_attr_cfg, input_attr_cfg)`` for ``method``, read from the stock
    named config. ``input_attr_cfg`` is ``None`` for weight-only methods."""
    named = _BACKBONE_CFG_MAP[method]
    weight_cfg: Optional[Dict[str, Any]] = None
    input_cfg: Optional[Dict[str, Any]] = None
    for entry in named["quant_cfg"]:
        name = entry.get("quantizer_name")
        if name == "*weight_quantizer" and "cfg" in entry:
            weight_cfg = entry["cfg"]
        elif name == "*input_quantizer" and "cfg" in entry:
            input_cfg = entry["cfg"]
    if weight_cfg is None:
        raise ValueError(
            f"ModelOpt config for {method!r} has no '*weight_quantizer' cfg.")
    return weight_cfg, input_cfg


def _enable_entries(method: str, weight_pattern: str,
                    input_pattern: str) -> List[QuantCfgEntry]:
    """Enable entries for one submodule, precision derived from ``method``. The
    input entry is skipped for weight-only methods."""
    weight_cfg, input_cfg = _method_weight_input_cfgs(method)
    entries: List[QuantCfgEntry] = [{
        "quantizer_name": weight_pattern,
        "cfg": copy.deepcopy(weight_cfg),
        "enable": True,
    }]
    if input_cfg is not None:
        entries.append({
            "quantizer_name": input_pattern,
            "cfg": copy.deepcopy(input_cfg),
            "enable": True,
        })
    return entries


def _disable_entries(patterns) -> List[QuantCfgEntry]:
    """``enable: False`` entries for each glob in ``patterns``."""
    return [{"quantizer_name": p, "enable": False} for p in patterns]


def _cp_entries(method: str) -> List[QuantCfgEntry]:
    """CodePredictor override: per-channel FP8 weight (axis=0) + per-tensor
    static input, minus the down_proj / lm_head / q,k,v-bmm submodules. Explicit
    (not derived) because the per-channel weight axis differs from the backbone.
    Disables come after the enables so the excludes win."""
    if method not in _CP_METHODS:
        raise ValueError(f"Unsupported cp_quantization: {method}. "
                         f"Choose from: {list(_CP_METHODS)}")
    weight_cfg = {"num_bits": (4, 3), "axis": 0}
    input_cfg = {"num_bits": (4, 3), "axis": None}
    out: List[QuantCfgEntry] = []
    for prefix in _CP_PREFIXES:
        out.append({
            "quantizer_name": f"*{prefix}*weight_quantizer",
            "cfg": copy.deepcopy(weight_cfg),
            "enable": True,
        })
        out.append({
            "quantizer_name": f"*{prefix}*input_quantizer",
            "cfg": copy.deepcopy(input_cfg),
            "enable": True,
        })
    for prefix in _CP_PREFIXES:
        for sub in _CP_LINEAR_EXCLUDES:
            out.append({
                "quantizer_name": f"*{prefix}*{sub}*weight_quantizer",
                "enable": False,
            })
            out.append({
                "quantizer_name": f"*{prefix}*{sub}*input_quantizer",
                "enable": False,
            })
        for sub in _CP_BMM_EXCLUDES:
            out.append({
                "quantizer_name": f"*{prefix}*{sub}_quantizer",
                "enable": False,
            })
    return out


def build_quant_config(
    quantization: Optional[str] = None,
    lm_head_quantization: Optional[str] = None,
    kv_cache_quantization: Optional[str] = None,
    visual_quantization: Optional[str] = None,
    visual_mha_quantization: Optional[str] = None,
    audio_quantization: Optional[str] = None,
    cp_quantization: Optional[str] = None,
    fuse_gdn_qkvzba_scales: bool = False,
) -> Dict[str, Any]:
    """Build a composite ModelOpt quantization config from method names.

    Returns ``{"quant_cfg": [<ordered rules>], "algorithm": ...}`` for
    :func:`mtq.quantize`. Overrides are appended on top of the stock backbone
    config and rely on last-match-wins ordering.

    Args:
        quantization:          Backbone precision (``fp8`` / ``nvfp4`` /
                               ``int4_awq`` / ``mxfp8`` / ``int8_sq``); ``None``
                               disables all quantizers.
        lm_head_quantization:  LM-head precision (backbone disables the head).
        kv_cache_quantization: KV-cache precision (``fp8`` only).
        visual_quantization:   Visual-tower precision; ``None`` leaves it off.
        visual_mha_quantization: Visual-attention (ViT MHA) precision;
                               ``fp8`` enables the q/k/v bmm quantizers so
                               calibration produces per-tensor dequant
                               scales. Orthogonal to visual_quantization.
        audio_quantization:    Audio-tower precision; ``None`` leaves it off.
        cp_quantization:       Qwen3-Omni CodePredictor precision; ``None``
                               leaves it off.
        fuse_gdn_qkvzba_scales: Re-enable NVFP4 quantization of the GDN
                               ``in_proj_b``/``in_proj_a`` projections (which
                               stock ModelOpt >=0.45 configs disable for
                               kernel-tiling compatibility on other backends).
                               Their per-tensor scales are then shared with
                               ``in_proj_qkv`` post-calibration so export can
                               fuse all four projections into a single GEMM.
    """
    if quantization is None:
        cfg: Dict[str, Any] = {
            "quant_cfg": [{
                "quantizer_name": "*",
                "enable": False
            }],
            "algorithm": "max",
        }
    elif quantization in _BACKBONE_CFG_MAP:
        cfg = copy.deepcopy(_BACKBONE_CFG_MAP[quantization])
    else:
        raise ValueError(f"Unsupported quantization: {quantization}. "
                         f"Choose from: {list(_BACKBONE_CFG_MAP)}")

    # KV cache: fold ``[k,v]_bmm`` in via the native helper, then add ``q_bmm``
    # so query BMM is quantized too.
    if kv_cache_quantization == "fp8":
        cfg = mtq.update_quant_cfg_with_kv_cache_quant(
            cfg, copy.deepcopy(mtq.FP8_KV_CFG["quant_cfg"]))
    elif kv_cache_quantization is not None:
        raise ValueError(
            f"Unsupported kv_cache_quantization: {kv_cache_quantization}. "
            "Choose from: ['fp8']")

    entries: List[QuantCfgEntry] = cfg["quant_cfg"]

    if kv_cache_quantization == "fp8":
        entries.append({
            "quantizer_name": "*q_bmm_quantizer",
            "cfg": {
                "num_bits": (4, 3),
                "axis": None
            },
            "enable": True,
        })

    # LM head: backbone disables it; re-enable at the requested precision.
    if lm_head_quantization is not None:
        if lm_head_quantization not in _LM_HEAD_METHODS:
            raise ValueError(
                f"Unsupported lm_head_quantization: {lm_head_quantization}. "
                f"Choose from: {list(_LM_HEAD_METHODS)}")
        entries += _enable_entries(lm_head_quantization,
                                   "*lm_head.weight_quantizer",
                                   "*lm_head.input_quantizer")

    # Disable every non-LLM group not opted into. code2wav is always off.
    disable_patterns: List[str] = list(_CODE2WAV_PATTERNS)
    if cp_quantization is None:
        disable_patterns += _CP_PATTERNS
    if visual_quantization is None:
        disable_patterns += _VISUAL_PATTERNS
    if audio_quantization is None:
        disable_patterns += _AUDIO_PATTERNS
    entries += _disable_entries(disable_patterns)

    # Re-enable opted-in modalities at the requested precision (after the
    # disables so they win).
    if visual_quantization is not None:
        if visual_quantization not in _VISUAL_METHODS:
            raise ValueError(
                f"Unsupported visual_quantization: {visual_quantization}. "
                f"Choose from: {list(_VISUAL_METHODS)}")
        for prefix in _VISUAL_PREFIXES:
            entries += _enable_entries(visual_quantization,
                                       f"*{prefix}*weight_quantizer",
                                       f"*{prefix}*input_quantizer")
        # Embedding tables inside visual towers (Qwen3-VL ``visual.pos_embed``,
        # Phi-4mm ``...embeddings.position_embedding``) match the prefix glob
        # but have no FP8 export/runtime path — keep them fp16 (disable after
        # enable so it wins).
        entries += _disable_entries([
            pat for prefix in _VISUAL_PREFIXES
            for pat in (f"*{prefix}*pos_embed*",
                        f"*{prefix}*position_embedding*")
        ])

    # Visual attention (ViT FP8 MHA): enable the q/k/v bmm quantizers on the
    # visual prefixes so calibration produces the per-tensor dequant scales the
    # ViT attention plugin consumes.
    if visual_mha_quantization is not None:
        if visual_mha_quantization != "fp8":
            raise ValueError(
                f"Unsupported visual_mha_quantization: "
                f"{visual_mha_quantization}. Choose from: ['fp8']")
        for prefix in _VISUAL_PREFIXES:
            for q in ("q", "k", "v"):
                entries.append({
                    "quantizer_name": f"*{prefix}*{q}_bmm_quantizer",
                    "cfg": {
                        "num_bits": (4, 3),
                        "axis": None
                    },
                    "enable": True,
                })

    if audio_quantization is not None:
        if audio_quantization not in _AUDIO_METHODS:
            raise ValueError(
                f"Unsupported audio_quantization: {audio_quantization}. "
                f"Choose from: {list(_AUDIO_METHODS)}")
        for prefix in _AUDIO_PREFIXES:
            entries += _enable_entries(audio_quantization,
                                       f"*{prefix}*weight_quantizer",
                                       f"*{prefix}*input_quantizer")

    # CP override last so its excludes win over the generic enables and q_bmm.
    if cp_quantization is not None:
        entries += _cp_entries(cp_quantization)

    # GDN b/a re-enable last so it wins over the stock config's disables.
    if fuse_gdn_qkvzba_scales:
        if quantization != "nvfp4":
            raise ValueError(
                "fuse_gdn_qkvzba_scales requires --quantization nvfp4, got "
                f"{quantization!r}")
        for proj in ("in_proj_b", "in_proj_a"):
            entries += _enable_entries(
                "nvfp4", f"*linear_attn.{proj}.weight_quantizer",
                f"*linear_attn.{proj}.input_quantizer")

    return cfg
