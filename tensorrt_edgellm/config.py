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
Model configuration parser for quantized LLM checkpoints.

Architecture fields are loaded with HuggingFace ``AutoConfig`` via
:func:`checkpoint_utils.load_checkpoint_config_dicts` (root + promoted LLM
dict; VL nested promotion applies to the LLM dict).  Quantization is merged from
``hf_quant_config.json`` or
embedded ``quantization_config`` via helpers in this module.

All model-specific feature flags (``attention_bias``, ``head_dim``, etc.)
are read from the resulting config dict.  The ``has_qk_norm`` flag is
auto-detected from the presence of ``q_norm`` weight keys in the checkpoint
index; no model-type string comparisons are used.

Supported quantization formats
-------------------------------
    fp16   - plain bfloat16/float16 weights (no quantization)
    fp8    - FP8 E4M3 per-tensor static quantization
    nvfp4  - NVFP4 per-group quantization with FP8 group scales
    nvfp4_a16 - weight-only NVFP4 (W4A16 / ModelOpt ``W4A16_NVFP4``)
    int4_awq            - AWQ INT4 group quantization (column-packed int32 checkpoints)
    int4_awq_modelopt   - W4A16_AWQ pre-packed uint8 ``[out//2, in]`` checkpoints
    int4_gptq           - GPTQ INT4 group quantization
    int8_gptq           - GPTQ/compressed-tensors INT8 weight-only quantization
    int8_sq             - INT8 SmoothQuant W8A8 per-channel
    mixed_precision     - per-layer mixed quantization (from ``hf_quant_config``)

Hybrid model support
--------------------
When ``config.json`` contains a ``layers_block_type`` list (e.g.
``["attention", "attention", "mamba", ...]``), a ``MambaConfig`` is parsed
and ``ModelConfig.layer_types`` reflects the per-layer block type.  Mamba
parameters (``mamba_num_heads``, ``mamba_head_dim``, ``ssm_state_size``,
``conv_dim``, ``conv_kernel``) are read from the config.  When ``conv_dim``
is absent (e.g. NemotronH-4B-BF16), it is derived from the ``conv1d.weight``
shape in the checkpoint to break the circular dependency with ``n_groups``.
"""

import fnmatch
import json
import math
import os
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import torch

from .checkpoint.checkpoint_utils import load_checkpoint_config_dicts
from .checkpoint.component_prefixes import VISUAL_PREFIXES
from .dflash import DFlashVersion, resolve_dflash_contract

# ---------------------------------------------------------------------------
# Quantization type constants
# ---------------------------------------------------------------------------

QUANT_FP16 = "fp16"
QUANT_FP8 = "fp8"
QUANT_MXFP8 = "mxfp8"
QUANT_NVFP4 = "nvfp4"
# Weight-only NVFP4 (W4A16): ModelOpt ``W4A16_NVFP4`` — 4-bit float weights,
# FP16 activations. Distinct from QUANT_NVFP4 (W4A4); routed to the dense/MoE
# Marlin FP16xE2M1 kernels.
QUANT_NVFP4_A16 = "nvfp4_a16"
QUANT_INT4_AWQ = "int4_awq"
QUANT_INT4_AWQ_MODELOPT = "int4_awq_modelopt"
QUANT_INT4_GPTQ = "int4_gptq"
QUANT_INT8_GPTQ = "int8_gptq"
QUANT_INT8_SQ = "int8_sq"
QUANT_Q8_0 = "q8_0"
QUANT_IQ2_XXS = "iq2_xxs"
# ``quant_algo`` value in hf_quant_config for MIXED_PRECISION only.
# After parsing, :attr:`QuantConfig.quant_type` is always a concrete type
# (dominant algo); per-layer differences live in ``layer_overrides``.
QUANT_MIXED = "mixed_precision"

# Default RoPE base frequency (used when config omits rope_theta)
_DEFAULT_ROPE_THETA = 10000.0

# Model families whose per-head QK RMSNorm runs AFTER RoPE (HunYuan V1);
# the default (Qwen3 convention) normalizes before rotation.
_QK_NORM_POST_ROPE_MODEL_TYPES = frozenset({"hunyuan_v1_dense"})

# Model families that store the per-head QK norms under the HunYuan
# ``query_layernorm`` / ``key_layernorm`` key names. Kept in sync with the
# ``_hunyuan_key_remap`` dispatch in model.py — QK-norm detection must not
# fire for families whose loader would not remap these keys onto q/k_norm.
_QUERY_LAYERNORM_KEY_MODEL_TYPES = frozenset({"hunyuan_v1_dense"})

# Layer-type labels
LAYER_ATTN = "attention"
LAYER_MAMBA = "mamba"
LAYER_MLP = "mlp"
LAYER_GDN = "gdn"  # GatedDeltaNet linear attention (Qwen3.5)
LAYER_MOE = "moe"

_VALID_ATTENTION_LAYER_TYPES = ("sliding_attention", "full_attention")

# NemotronH ``hybrid_override_pattern`` / ``mtp_hybrid_override_pattern`` chars.
_HYBRID_PATTERN_MAP = {
    "M": LAYER_MAMBA,
    "-": LAYER_MLP,
    "*": LAYER_ATTN,
    "E": LAYER_MOE,
}

_DIFFUSION_GEMMA_MODEL_TYPES = frozenset({
    "diffusion_gemma",
    "diffusion_gemma_text",
    "diffusiongemma",
})
_DEFAULT_DIFFUSION_MAX_DENOISING_STEPS = 48


def _is_diffusion_gemma_model_type(model_type: str) -> bool:
    return str(model_type).lower() in _DIFFUSION_GEMMA_MODEL_TYPES


def _is_diffusion_gemma_config(root_dict: Dict[str, Any],
                               llm_dict: Dict[str, Any]) -> bool:
    model_type = str(
        root_dict.get("model_type", llm_dict.get("model_type", ""))).lower()
    architectures_value = (root_dict.get("architectures")
                           or llm_dict.get("architectures") or [])
    if isinstance(architectures_value, str):
        architectures = [architectures_value]
    else:
        architectures = [str(x) for x in architectures_value]
    return (_is_diffusion_gemma_model_type(model_type)
            or any("DiffusionGemma" in arch for arch in architectures))


def _is_gemma4_model_type(model_type: str) -> bool:
    model_type = str(model_type)
    return (model_type.startswith("gemma4")
            or _is_diffusion_gemma_model_type(model_type))


def _is_gemma4_assistant_model_type(model_type: str) -> bool:
    return str(model_type) in ("gemma4_assistant", "gemma4_unified_assistant")


def _is_muse_glimmer_model_type(model_type: str) -> bool:
    """Return whether a model type belongs to Muse-Glimmer."""
    return str(model_type) in ("muse_glimmer", "muse_glimmer_text")


def _check_num_attention_heads(num_attn_heads: int) -> None:
    if num_attn_heads <= 0:
        raise ValueError("num_attention_heads must be a positive integer, "
                         f"got {num_attn_heads!r}")


def _get_rope_theta(llm_dict: Dict[str, Any]) -> float:
    """Extract rope_theta from config dict.

    Some models (e.g. Qwen3) store rope_theta inside ``rope_scaling`` or
    ``rope_parameters`` rather than as a top-level key, so fall back to those
    nested dicts before returning the default 10 000.
    """
    if llm_dict.get("rope_theta") is not None:
        return float(llm_dict["rope_theta"])
    for key in ("rope_scaling", "rope_parameters"):
        nested = llm_dict.get(key)
        if not isinstance(nested, dict):
            continue
        if nested.get("rope_theta") is not None:
            return float(nested["rope_theta"])
        for attention_type in ("full_attention", "sliding_attention"):
            attention_params = nested.get(attention_type)
            if (isinstance(attention_params, dict)
                    and attention_params.get("rope_theta") is not None):
                return float(attention_params["rope_theta"])
    return _DEFAULT_ROPE_THETA


def _normalize_rope_scaling_for_config(
        rope_params: Optional[Dict[str, Any]]) -> Optional[dict]:
    """Normalize one RoPE parameter block for runtime config export."""
    if not isinstance(rope_params, dict):
        return None
    normalized = dict(rope_params)
    rope_type = normalized.get("rope_type", normalized.get("type"))
    if normalized.get("mrope_section") is not None:
        normalized["rope_type"] = "mrope"
        normalized["type"] = "mrope"
    elif rope_type is not None:
        normalized.setdefault("rope_type", rope_type)
        normalized.setdefault("type", rope_type)
    return normalized


def _select_rope_scaling(llm_dict: Dict[str, Any]) -> Optional[dict]:
    """Select the single-RoPE fallback block from raw config metadata."""
    for key in ("rope_scaling", "rope_parameters"):
        nested = llm_dict.get(key)
        if not isinstance(nested, dict):
            continue
        if isinstance(nested.get("full_attention"), dict):
            return _normalize_rope_scaling_for_config(nested["full_attention"])
        if isinstance(nested.get("sliding_attention"), dict):
            return _normalize_rope_scaling_for_config(
                nested["sliding_attention"])
        return _normalize_rope_scaling_for_config(nested)
    return None


def _runtime_rope_config_from_params(llm_dict: Dict[str, Any],
                                     rope_params: Dict[str, Any]) -> dict:
    """Build one runtime RoPE config block from raw checkpoint metadata."""
    out = {
        "rope_theta":
        float(
            rope_params.get("rope_theta",
                            llm_dict.get("rope_theta", _DEFAULT_ROPE_THETA))),
        "rope_scaling":
        _normalize_rope_scaling_for_config(rope_params),
        "partial_rotary_factor":
        float(
            rope_params.get("partial_rotary_factor",
                            llm_dict.get("partial_rotary_factor", 1.0))),
        "max_position_embeddings":
        int(llm_dict.get("max_position_embeddings", 4096)),
    }
    return out


def _get_dual_rope_configs(llm_dict: Dict[str, Any]) -> dict[str, dict]:
    """Return explicit sliding/full runtime RoPE configs when present."""
    rope_parameters = llm_dict.get("rope_parameters")
    layer_types = {
        str(layer_type)
        for layer_type in llm_dict.get("layer_types", [])
    }
    if not isinstance(rope_parameters, dict):
        return {}
    if not {"sliding_attention", "full_attention"} <= layer_types:
        return {}

    sliding_params = rope_parameters.get("sliding_attention")
    full_params = rope_parameters.get("full_attention")
    if not isinstance(sliding_params, dict) or not isinstance(
            full_params, dict):
        return {}
    return {
        "sliding_rope_config":
        _runtime_rope_config_from_params(llm_dict, sliding_params),
        "full_rope_config":
        _runtime_rope_config_from_params(llm_dict, full_params),
    }


def _parse_attention_layer_types(config: dict, num_hidden_layers: int,
                                 model_type: str) -> List[str]:
    """Preserve per-layer sliding/full attention labels when required."""
    raw = config.get("layer_types")
    if not (_is_gemma4_model_type(model_type)
            or _is_muse_glimmer_model_type(model_type)):
        return []

    if not isinstance(raw, list):
        raise ValueError(
            f"{model_type} requires layer_types with one sliding/full "
            "attention entry per layer.")
    if len(raw) != num_hidden_layers:
        raise ValueError(
            f"{model_type} layer_types length must match num_hidden_layers: "
            f"{len(raw)} vs {num_hidden_layers}.")

    attention_layer_types: List[str] = []
    for layer_idx, layer_type in enumerate(raw):
        layer_type = str(layer_type)
        if layer_type not in _VALID_ATTENTION_LAYER_TYPES:
            raise ValueError(
                f"{model_type} layer_types entries must be sliding_attention "
                "or full_attention; "
                f"got {layer_type!r} at layer {layer_idx}.")
        attention_layer_types.append(layer_type)
    return attention_layer_types


def _get_attention_scaling(llm_dict: Dict[str, Any], head_dim: int,
                           default_val: float) -> float:
    """Resolve the absolute multiplier applied to QK^T before softmax.

    Args:
        llm_dict: Checkpoint configuration containing an optional supported
            attention-scale alias.
        head_dim: Per-head query/key dimension.
        default_val: Required model-family fallback.

    Returns:
        The finite, positive attention scale supplied by the checkpoint, or
        the caller-selected fallback.

    Raises:
        ValueError: If an explicit checkpoint scale is not finite and
            positive, or if ``head_dim`` is not positive.
    """
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive; got {head_dim}")

    for key in ("attention_scaling", "qk_scale", "scaling"):
        if llm_dict.get(key) is not None:
            attention_scale = float(llm_dict[key])
            if not math.isfinite(attention_scale) or attention_scale <= 0.0:
                raise ValueError(
                    f"{key} must be finite and positive; got {llm_dict[key]!r}"
                )
            return attention_scale

    attention_scale = float(default_val)
    if not math.isfinite(attention_scale) or attention_scale <= 0.0:
        raise ValueError(
            f"default_val must be finite and positive; got {default_val!r}")
    return attention_scale


def _get_rms_norm_eps(llm_dict: Dict[str, Any], model_type: str) -> float:
    if str(model_type).lower().startswith("nemotron_h"):
        return llm_dict.get(
            "rms_norm_eps",
            llm_dict.get("norm_eps", llm_dict.get("layer_norm_epsilon", 1e-6)))

    return llm_dict.get("rms_norm_eps", 1e-6)


def _get_embedding_scale(llm_dict: Dict[str, Any], model_type: str,
                         hidden_size: int) -> float:
    """Return the scale folded into runtime token embeddings."""
    for key in ("embedding_scale", "embed_scale", "scalar_embed_scale"):
        if llm_dict.get(key) is not None:
            return float(llm_dict[key])

    if _is_gemma4_model_type(model_type):
        return math.sqrt(float(hidden_size))

    return 1.0


def _get_has_value_norm(llm_dict: Dict[str, Any], model_type: str) -> bool:
    """Return whether attention values use Gemma-style RMSNorm."""
    for key in ("has_value_norm", "has_v_norm", "value_norm"):
        if llm_dict.get(key) is not None:
            return bool(llm_dict[key])

    return _is_gemma4_model_type(model_type)


@dataclass
class Mapping:
    """Tensor-parallel placement for one exported or loaded rank.

    ``tp_size`` / ``tp_rank`` are the only supported model-sharding
    coordinates in the current Edge-LLM multi-device flow. Future CP/EP/PP/DP
    work should add its own mapping fields together with matching export,
    runtime, and validation support.
    """
    world_size: int = 1
    rank: int = 0
    tp_size: int = 1
    tp_rank: int = 0


@dataclass
class ActionConfig:
    """Action expert hyper-parameters for Alpamayo models.

    Used only for the action expert ONNX export and its sidecar config.json.
    These fields do NOT feed into the LLM config.json.
    """

    rope_theta: float = 5_000_000.0
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])
    mrope_interleaved: bool = True
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 128
    attention_scaling: Optional[float] = None
    hidden_size: int = 0
    intermediate_size: int = 0
    rms_norm_eps: float = 1e-6
    num_traj_tokens: int = 1000
    traj_token_start: int = 0
    n_diffusion_tokens: int = 64
    in_proj_hidden_size: int = 512
    in_proj_num_enc_layers: int = 2
    in_proj_max_freq: float = 100.0
    in_proj_num_fourier_feats: int = 20
    kv_cache_scales: Optional[List[List[float]]] = None

    def __post_init__(self) -> None:
        """Resolve and validate attention scaling for direct construction.

        Raises:
            ValueError: If the head dimension or explicit scale is invalid.
        """
        scale_config = ({} if self.attention_scaling is None else {
            "attention_scaling": self.attention_scaling
        })
        self.attention_scaling = _get_attention_scaling(
            scale_config, self.head_dim, 1.0 / (float(self.head_dim)**0.5))


_DEFAULT_QUANTIZE_ACTIVATIONS = True


def set_default_quantize_activations(value: bool) -> None:
    """Set :attr:`QuantConfig.quantize_activations` for configs parsed later.

    The export CLI calls this once from its argument parsing. A whole export can
    build several QuantConfigs (backbone plus any draft model), all of them
    parsed from their checkpoints afterwards, so one assignment covers the run
    without threading the flag through every ``_export_*`` entry point.
    """
    global _DEFAULT_QUANTIZE_ACTIVATIONS
    _DEFAULT_QUANTIZE_ACTIVATIONS = value


@dataclass
class QuantConfig:
    """Quantization parameters extracted from the checkpoint config."""

    quant_type: str = QUANT_FP16
    # group_size: 1 = per-tensor/per-channel, 16 for NVFP4, 128 for AWQ
    group_size: int = 1
    # GPTQ checkpoints are not consistent about whether qzeros stores the
    # actual zero point or (zero point - 1).  The loader uses:
    # actual_zero = stored_zero + gptq_zero_point_offset.
    gptq_zero_point_offset: int = 1
    # kv_cache_quant: "fp8" when KV-cache is quantised, None otherwise
    kv_cache_quant: Optional[str] = None
    # visual_mha_quant: "fp8" when the ViT visual attention (Q*K^T / P*V
    # matmuls) is quantised, None otherwise. Orthogonal to kv_cache_quant
    # (which only gates the LLM KV cache).
    visual_mha_quant: Optional[str] = None
    # module names excluded from quantisation (typically ["lm_head"])
    excluded: List[str] = field(default_factory=list)
    # Per-layer quant type overrides for MIXED_PRECISION checkpoints.
    # Maps a module name (e.g. "lm_head") to a quant-type string.
    # make_linear() uses module_name together with ``excluded`` and (for
    # lm_head) ``ModelConfig.tie_word_embeddings`` to pick FP16 vs overrides.
    layer_overrides: dict = field(default_factory=dict)
    # Per-layer group sizes use the same normalized names as layer_overrides.
    layer_group_sizes: dict = field(default_factory=dict)
    # True when quant_algo is MIXED_PRECISION: unlisted modules are FP16.
    is_mixed_precision: bool = False
    # False selects the NVFP4-A16 kernel for packed NVFP4 weights. Other
    # activation-quantized formats omit their activation Q-DQ pair for accuracy
    # debugging. The default reproduces the checkpoint's W4A4/W8A8 recipe.
    quantize_activations: bool = field(
        default_factory=lambda: _DEFAULT_QUANTIZE_ACTIVATIONS)

    @property
    def is_quantized(self) -> bool:
        return self.quant_type != QUANT_FP16

    @property
    def uses_nvfp4_weights(self) -> bool:
        """True if any linear uses NVFP4 weights (dominant quant or layer override)."""
        nvfp4_types = (QUANT_NVFP4, QUANT_NVFP4_A16)
        if self.quant_type in nvfp4_types:
            return True
        return any(v in nvfp4_types for v in self.layer_overrides.values())

    @property
    def uses_mxfp8_weights(self) -> bool:
        """True if any linear uses MXFP8 weights (dominant quant or layer override)."""
        if self.quant_type == QUANT_MXFP8:
            return True
        return any(v == QUANT_MXFP8 for v in self.layer_overrides.values())


def module_quant_type(module_name: str, model_config: "ModelConfig") -> str:
    """Return the effective quant type ``make_linear`` will pick for *module_name*.

    Single source of truth for "what precision does this module's Linear end
    up at?" — used by :func:`make_linear` to pick the Linear class, and by
    the ONNX exporter to validate that LM-head externalization is only
    requested for an fp16 head.  Lookup matches the names used elsewhere:
    ``ModelConfig.quant.excluded`` and ``layer_overrides`` keys are already
    normalized (VL prefixes / ``model.`` stripped) when parsed, and callers
    pass the same short ``module_name`` that ``make_linear`` receives
    (e.g. ``"lm_head"``).
    """
    quant = model_config.quant
    if module_name and any(
            fnmatch.fnmatchcase(module_name, pattern)
            for pattern in quant.excluded):
        return QUANT_FP16
    # Tied lm_head with no explicit override and an unquantized backbone has
    # no separate lm_head.weight in the checkpoint; treat it as fp16 so the
    # weight can be cloned from embed_tokens after loading.
    if (module_name == "lm_head" and model_config.tie_word_embeddings
            and "lm_head" not in quant.layer_overrides
            and quant.quant_type == QUANT_FP16):
        return QUANT_FP16
    quant_type = quant.quant_type
    if module_name and quant.layer_overrides:
        fallback = QUANT_FP16 if quant.is_mixed_precision else quant_type
        quant_type = quant.layer_overrides.get(module_name, fallback)
    return quant_type


def module_quant_group_size(module_name: str,
                            model_config: "ModelConfig") -> int:
    """Return the checkpoint group size for a quantized linear module."""
    quant = model_config.quant
    return int(quant.layer_group_sizes.get(module_name, quant.group_size))


@dataclass
class MambaConfig:
    """Mamba-layer hyper-parameters for hybrid models."""

    num_heads: int  # mamba_num_heads
    head_dim: int  # mamba_head_dim
    ssm_state_size: int  # ssm_state_size
    conv_dim: int  # conv_dim (total convolution channel count)
    conv_kernel: int  # conv1d kernel size (default 4)
    n_groups: int  # number of SSM groups (derived if not in config)

    @property
    def intermediate_size(self) -> int:
        """num_heads x head_dim - the Mamba intermediate feature dimension."""
        return self.num_heads * self.head_dim


@dataclass
class GdnConfig:
    """GatedDeltaNet (GDN) hyper-parameters for Qwen3.5 hybrid models.

    GDN layers use a gated delta-net linear attention mechanism with
    fused QKV projection through causal conv1d.
    """

    num_key_heads: int  # linear_num_key_heads
    num_value_heads: int  # linear_num_value_heads
    key_head_dim: int  # linear_key_head_dim
    value_head_dim: int  # linear_value_head_dim
    conv_kernel: int  # linear_conv_kernel_dim (default 4)
    output_gate_type: str  # output_gate_type or hidden_act (ACT2FN name)

    @property
    def key_dim(self) -> int:
        """Total key dimension (num_key_heads * key_head_dim)."""
        return self.num_key_heads * self.key_head_dim

    @property
    def value_dim(self) -> int:
        """Total value dimension (num_value_heads * value_head_dim)."""
        return self.num_value_heads * self.value_head_dim

    @property
    def conv_dim(self) -> int:
        """Total conv1d channel count: key + key + value (QKV fused)."""
        return self.key_dim + self.key_dim + self.value_dim


@dataclass
class DiffusionConfig:
    """Block-diffusion generation parameters parsed at export time."""

    diffusion_family: str = "uniform_renoise"
    canvas_length: int = 256
    max_denoising_steps: int = _DEFAULT_DIFFUSION_MAX_DENOISING_STEPS
    t_max: float = 0.8
    t_min: float = 0.4
    sampler_type: str = "entropy_bound"
    entropy_bound: float = 0.1
    entropy_threshold: float = 0.005
    stability_window: int = 2
    self_conditioning_enabled: bool = True
    self_conditioning_repr: str = "embeds"
    supported_modalities: List[str] = field(default_factory=lambda: ["text"])

    @classmethod
    def from_hf(
            cls,
            hf_config: Dict[str, Any],
            gen_config: Optional[Dict[str, Any]] = None) -> "DiffusionConfig":
        gen_config = gen_config or {}
        sampler_cfg = gen_config.get("sampler_config", {}) or {}
        sampler_name = str(sampler_cfg.get("_cls_name", "EntropyBound"))
        sampler_type = ("entropy_bound"
                        if "entropybound" in sampler_name.lower().replace(
                            "_", "") else sampler_name.lower())
        return cls(
            canvas_length=int(
                hf_config.get("canvas_length",
                              gen_config.get("canvas_length", 256))),
            max_denoising_steps=int(
                gen_config.get("max_denoising_steps",
                               _DEFAULT_DIFFUSION_MAX_DENOISING_STEPS)),
            t_max=float(gen_config.get("t_max", 0.8)),
            t_min=float(gen_config.get("t_min", 0.4)),
            sampler_type=sampler_type,
            entropy_bound=float(
                sampler_cfg.get("entropy_bound",
                                gen_config.get("entropy_bound", 0.1))),
            entropy_threshold=float(
                gen_config.get("entropy_threshold",
                               gen_config.get("confidence_threshold", 0.005))),
            stability_window=int(
                gen_config.get("stability_window",
                               gen_config.get("stability_threshold", 2))),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    """Flat model hyper-parameter config consumed by module builders."""

    # ------------------------------------------------------------------ arch
    model_type: str  # HF architecture name, e.g. "qwen3", "llama"
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float
    max_position_embeddings: int
    default_attention_scale: float
    # RoPE scaling config (e.g. {"type": "dynamic", "factor": 2.0} for Qwen2).
    # None means no scaling (standard RoPE).
    rope_scaling: Optional[dict] = None
    # For longrope: original context window size before extension.
    original_max_position_embeddings: Optional[int] = None
    # Fraction of head_dim used for RoPE (e.g. 0.75 for phi3/phi4, 1.0 for most others).
    partial_rotary_factor: float = 1.0
    # Gemma4: head_dim for global (full_attention) layers (0 = same as head_dim)
    global_head_dim: int = 0
    # Gemma4 E4B: num_key_value_heads for global (full_attention) layers (0 = same as num_key_value_heads)
    num_global_key_value_heads: int = 0
    # QSA (Qwen Sparse Attention): indexer head dim; QSA layers widen their paged KV pool
    # rows by this many columns to persist indexer state (0 = no QSA layers).
    indexer_head_dim: int = 0
    # Hidden activation name used by architecture-specific auxiliary modules.
    hidden_activation: str = "silu"
    # CodePredictor: RVQ code groups (lm_heads count = num_code_groups - 1).
    num_code_groups: int = 0
    # Optional explicit RoPE configs for mixed sliding/full attention stacks.
    sliding_rope_config: Optional[dict] = None
    full_rope_config: Optional[dict] = None
    # ------------------------------------------ model-family feature flags
    # Per-head RMSNorm after Q and K projections.
    # Auto-detected from checkpoint key names; not inferred from model_type.
    has_qk_norm: bool = False
    # QK-norm order relative to RoPE. False (Qwen3 convention): norm then
    # rotate. True (HunYuan V1): rotate then norm — gamma placement differs,
    # so the attention plugin must apply the norm after rotation.
    qk_norm_post_rope: bool = False
    # Per-head RMSNorm after V projection. Gemma4 stores this norm without
    # learned weights, so it is selected from config metadata instead of
    # checkpoint key names.
    has_value_norm: bool = False
    # Bias on q/k/v projections.  Read from config.json "attention_bias".
    attention_bias: bool = False
    # Explicit multiplicative scale applied to QK^T before softmax.
    attention_scaling: Optional[float] = None
    # Gemma4 full/global attention can use a different per-head dimension
    # from sliding attention.
    global_head_dim: Optional[int] = None
    # Gemma4 K=V full/global attention can use a different KV head count from
    # sliding attention.
    num_global_key_value_heads: Optional[int] = None
    # Gemma4 full/global attention reuses k_proj(hidden_states) as the value
    # projection source when enabled.
    attention_k_eq_v: bool = False
    # Per-Q-head learned attention sink: extra logit merged into softmax denominator.
    attention_sink_bias: bool = False
    # DiffusionGemma uses one shared backbone with phase-dependent layer scalars.
    encoder_layer_scalars: List[float] = field(default_factory=list)
    decoder_layer_scalars: List[float] = field(default_factory=list)
    self_conditioning_size: int = 0
    diffusion: Optional[DiffusionConfig] = None
    # Multiplicative scale applied by the HF embedding module.
    embedding_scale: float = 1.0
    # Final logit softcapping: tanh(logits/cap)*cap.  None = disabled.
    final_logit_softcapping: Optional[float] = None
    # Muse-Glimmer: pre-tanh logit multiplier (applied before the softcap).
    output_multiplier: float = 1.0
    # Muse-Glimmer: eps for the post-attention / post-FFN sandwich norms
    # (None -> fall back to rms_norm_eps).
    post_norm_eps: Optional[float] = None
    # Weight dtype in the checkpoint
    torch_dtype: str = "bfloat16"
    # When True, embed_tokens and lm_head share the same weight tensor
    tie_word_embeddings: bool = False
    # Sliding window attention size; -1 means no sliding window.
    sliding_window_size: int = -1
    # Which skip-softmax (BLASST) scale tables the engine deploys; each wires
    # the matching plugin input (skip_softmax_prefill_scale /
    # skip_softmax_decode_scale). Set by the exporter from the resolved
    # deployment (tensorrt_edgellm.skip_softmax), never read from the checkpoint.
    skip_softmax_prefill_enabled: bool = False
    skip_softmax_decode_enabled: bool = False
    # Gemma4 Unified 12B+: image placeholder runs use block-causal
    # attention during prefill (bidirectional inside each contiguous vision
    # run, causal everywhere else).  Audio placeholders remain causal.
    use_vision_bidirectional_attention: bool = False
    # ------------------------------------------ per-layer block types
    # One entry per hidden layer: LAYER_ATTN, LAYER_MAMBA, LAYER_MLP, or LAYER_MOE.
    layer_types: List[str] = field(default_factory=list)
    # Nemotron-H Puzzle: one block definition per layer; MoE widths and
    # expert counts are not uniform across the decoder.
    block_configs: List[dict] = field(default_factory=list)
    # Original attention labels for attention layers: full_attention or sliding_attention.
    attention_layer_types: List[str] = field(default_factory=list)
    # ------------------------------------------ multimodal deepstack (VL)
    # Number of deepstack visual embedding tensors injected into the first N
    # hidden layers. Prefer ``vision_config.deepstack_visual_indexes`` length on
    # the root config when present; else ``num_deepstack_features`` or fallback.
    num_deepstack_features: int = 0
    # ----------------------------------------- Qwen3-Omni emitted-tensor layer
    # Read by the Qwen3-Omni-specific ``Transformer`` subclasses (dense and
    # MoE) to decide which tensor to expose via ``emitted_hidden_states``
    # (consumed by their ``CausalLM`` wrappers when
    # ``emit_hidden_states = True``):
    #
    # * ``accept_hidden_layer >= 1`` (and ≤ num_hidden_layers): pre-norm
    #   output of decoder layer ``accept_hidden_layer - 1``.  Matches HF's
    #   ``outputs.hidden_states[k]`` convention where ``k`` denotes "after
    #   ``k`` decoder layers" (k=0 is inputs_embeds, k=N is the last layer
    #   pre-norm).  Used by Qwen3-Omni Thinker → Talker: Talker
    #   consumes ``thinker.hidden_states[accept_hidden_layer]``.
    #
    # * Default ``-1``: post-final-norm output (= ``model.norm(last_layer)``).
    #   Used by Qwen3-Omni Talker → CodePredictor: HF reads
    #   ``hidden_states[0][-1]`` which resolves to the post-norm tensor.
    accept_hidden_layer: int = -1
    # -------------------------------------------------- quantization config
    quant: QuantConfig = field(default_factory=QuantConfig)
    # ------------------------------------------ mamba / hybrid config
    mamba_cfg: Optional[MambaConfig] = None
    # ------------------------------------------ gdn / hybrid config
    gdn_cfg: Optional[GdnConfig] = None
    # ------------------------------------------ gated attention (Qwen3.5)
    attn_output_gate: bool = False
    # ------------------------------------------ gated residual (Qwen4-Exp)
    # Parallel residual streams ("HyperConnections"); 0 = single residual bus.
    hc_count: int = 0
    # Bottleneck width of the per-stream read-gate projection.
    hc_lowrank: int = 0
    # ------------------------------------------ QSA indexer (Qwen4-Exp)
    # Block-compressed indexer picking which KV blocks sparse attention reads.
    # 0 = dense attention.  block_topk = indexer_budget // indexer_compress_ratio.
    indexer_n_heads: int = 0
    indexer_kv_heads: int = 0
    indexer_budget: int = 0
    indexer_compress_ratio: int = 0
    # ------------------------------------------ PLE (Qwen4-Exp)
    # Layers carrying an n-gram embedding injected into the residual, 1-indexed
    # as the reference config writes them. Empty for every checkpoint that does
    # not use PLE, which is what keeps the whole path optional.
    ple_layer_ids: List[int] = field(default_factory=list)
    # Width of the gathered n-gram vector the runtime hands to the engine.
    ple_embed_dim: int = 0
    # Kernel of the depthwise conv that widens the injection; its dilation is
    # ``ngram_size``, so its state spans ``(kernel - 1) * ngram_size`` steps.
    ple_conv_kernel_size: int = 0
    # Longest n-gram order hashed into a row ID, and the hash heads per order.
    # The runtime needs both to size the row-ID tensor and the token history.
    ngram_size: int = 0
    heads_per_ngram: int = 0
    # Row blocks the checkpoint stores the table in.
    split_ngram_parts: int = 0
    # ------------------------------------------ MTP config
    mtp_num_hidden_layers: Optional[int] = None
    mtp_use_dedicated_embeddings: bool = False
    # Nemotron-H MTP: the draft module is a hybrid stack whose layer types come
    # from this pattern (e.g. "*E" -> [attention, MoE]); ``mtp_num_hidden_layers``
    # is then its length. ``num_nextn_predict_layers`` (the count of MTP prediction
    # modules) is folded in during parsing.
    mtp_hybrid_override_pattern: Optional[str] = None
    # The same draft stack, resolved from either the pattern above or the
    # ``mtp_layers_block_type`` list.
    mtp_layer_types: List[str] = field(default_factory=list)
    # Set by make_mtp_draft_config; the base and its derived draft config are
    # otherwise indistinguishable by their fields for dense checkpoints.
    is_mtp_draft_flag: bool = False
    # When True, the standard CausalLM is exported as the MTP base model variant
    # with tree-attention inputs (attention_mask, attention_pos_id) and
    # an extra hidden_states output.
    mtp_base: bool = False
    # ------------------------------------------ Gemma4 MTP config
    root_model_type: str = ""
    raw_layer_types: List[str] = field(default_factory=list)
    rope_parameters: Optional[dict] = None
    backbone_hidden_size: int = 0
    gemma4_mtp_base: bool = False
    gemma4_mtp_draft: bool = False
    assistant_hidden_size: int = 0
    shares_target_kv: bool = False
    has_own_kv_cache: bool = True
    constant_draft_positions: bool = False
    returns_feedback_hidden: bool = False
    use_ordered_embeddings: bool = False
    num_centroids: int = 0
    centroid_intermediate_top_k: int = 0
    sparse_logits_enabled: bool = False
    kv_sharing_map: List[dict] = field(default_factory=list)
    # ------------------------------------------ EAGLE3 draft config
    draft_vocab_size: Optional[int] = None
    target_hidden_size: Optional[int] = None
    is_eagle3_draft_flag: bool = False
    # HF hidden-state indices (input of decoder layer k); Gemma4 bases read
    # flat draft IDs as decoder-layer outputs instead.
    eagle3_target_layer_ids: List[int] = field(default_factory=list)
    eagle3_norm_before_residual: bool = False
    eagle3_norm_before_fc: bool = False
    # Per-target-layer RMSNorm on the fc input.
    eagle_fc_norm: bool = False
    # The draft feeds back its final-norm output instead of the pre-norm one.
    eagle_norm_output: bool = False
    # Low-rank candidate head; 0 means the dense lm_head.
    draft_candidate_rank: int = 0
    draft_candidate_count: int = 0
    # ------------------------------------------ EAGLE3 base config
    # When True, the standard CausalLM is exported as an EAGLE3 base model
    # with tree-attention inputs (attention_mask, attention_pos_id) and
    # an extra hidden_states output (concatenated from 3 selected layers).
    eagle_base: bool = False
    # ------------------------------------------ DFlash config
    # When True, export the standard Qwen3.5 model as the DFlash base with
    # tree-attention verify inputs and multi-layer hidden_states output.
    dflash_base: bool = False
    is_dflash_draft_flag: bool = False
    dflash_target_layer_ids: List[int] = field(default_factory=list)
    dflash_block_size: int = 16
    dflash_mask_token_id: int = 248070
    dflash_version: DFlashVersion = DFlashVersion.V1
    dflash_overflow_fp32: bool = False
    # Run the fc feature projector at the checkpoint's native precision (e.g.
    # NVFP4) instead of the default dense-FP16 + FP32 projection. Enabled only
    # for targets measured to keep target-hidden well inside FP16 range
    # (Nemotron-3.5). Qwen3-8B keeps the FP32 guard (target-hidden ~abs 2e4).
    dflash_fc_native_precision: bool = False
    # DFlash2 is a distinct linear-path proposal architecture. Its checkpoint
    # block size is the runtime default; an engine may profile a larger block.
    dflash2_target_layer_ids: List[int] = field(default_factory=list)
    dflash2_block_size: int = 8
    dflash2_mask_token_id: int = 248070
    dflash2_is_causal: bool = False
    dflash2_conv_kernel_size: int = 2
    dflash2_conv_group_size: int = 16
    dflash2_selector_rank: int = 256
    dflash2_selector_top_k: int = 16
    dflash2_output_multiplier: float = 1.0
    dflash2_final_logit_softcapping: float = 0.0
    # ------------------------------------------ Domino config
    # Domino keeps the DFlash V1 draft ABI and adds a runtime causal
    # correction head (GRU over realized-token embeddings + a rank-proj_dim
    # vocab codebook). The candidate pool (top-pool_size tokens by base
    # logits over the feedback positions) is materialized inside the draft
    # engine; the GRU and state projection run as runtime sidecars.
    dflash_projector_type: str = ""
    domino_gru_hidden_dim: int = 1024
    domino_proj_dim: int = 256
    domino_prefix_len: int = 1
    domino_pool_size: int = 2048
    domino_shift_label: bool = False
    # LiLiCorr (dflash_projector_type == "lilicorr"): per-slot candidate count.
    lilicorr_candidate_topk: int = 0
    # LiLiCorr lattice head geometry; logit_scale multiplies cosine scores.
    lilicorr_hidden_size: int = 0
    lilicorr_num_layers: int = 0
    lilicorr_num_heads: int = 0
    lilicorr_mlp_ratio: float = 0.0
    lilicorr_factor_dim: int = 0
    lilicorr_vector_eps: float = 0.0
    lilicorr_logit_scale: float = 0.0
    # DFlash V1 draft whose block position i predicts token i + 1 (DeepSpec).
    dflash_shift_label: bool = False
    # ------------------------------------------ JetSpec config
    # JetSpec uses the DFlash/DDTree cached-draft contract with causal proposal
    # attention inside the draft block. The DFlash-prefixed fields are still
    # populated for the shared DFlashDraftModel ONNX implementation.
    jetspec_base: bool = False
    is_jetspec_draft_flag: bool = False
    jetspec_target_layer_ids: List[int] = field(default_factory=list)
    jetspec_block_size: int = 16
    jetspec_mask_token_id: int = 151669
    jetspec_causal_head: bool = False
    # ------------------------------------------ DSpark config
    # DSpark uses the DFlash-like target-hidden feedback path, then applies
    # a sequential Markov/confidence head outside the draft backbone engine.
    dspark_base: bool = False
    is_dspark_draft_flag: bool = False
    dspark_target_layer_ids: List[int] = field(default_factory=list)
    dspark_block_size: int = 7
    dspark_mask_token_id: int = 151669
    dspark_enable_confidence_head: bool = False
    dspark_sample_from_anchor: bool = True
    dspark_confidence_head_with_markov: bool = False
    dspark_markov_head_type: str = ""
    dspark_markov_rank: int = 0
    dspark_fc_native_precision: bool = False
    dspark_causal_proposal: bool = False
    # ------------------------------------------ sparse MoE config (Qwen3-style)
    # num_experts=0 means dense (no MoE) for Qwen/Mixtral-style keys; Nemotron-H instead reports
    # its expert count via n_routed_experts, so n_routed_experts > 0 also indicates MoE.
    num_experts: int = 0
    n_routed_experts: int = 0
    num_experts_per_tok: int = 0
    # Expert MLP intermediate size (may differ from dense intermediate_size).
    moe_intermediate_size: int = 0
    moe_shared_expert_intermediate_size: int = 0
    # Optional latent dimension used by Nemotron-H routed experts.
    moe_latent_size: Optional[int] = None
    routed_scaling_factor: float = 1.0
    n_group: int = 1
    topk_group: int = 1
    # MoE layer frequency: layer (i+1) % decoder_sparse_step == 0 → MoE.
    decoder_sparse_step: int = 1
    # Layer indices that are always dense MLP (overrides decoder_sparse_step).
    mlp_only_layers: List[int] = field(default_factory=list)
    # Normalise top-k routing weights to sum to 1.
    # Note: the C++ Int4MoePlugin hardcodes renormalize=true, so this field
    # currently serves as documentation of the HF config value.
    norm_topk_prob: bool = True
    # Runtime vocabulary reduction. ``vocab_size`` remains the original
    # tokenizer/embedding size; this field is only the exported logits size.
    reduced_vocab_size: Optional[int] = None
    # ------------------------------------------ per-layer embeddings (Gemma4)
    # When > 0, Gemma4 E-model PLE is enabled. The ONNX graph receives
    # a shared row pool and token-to-slot map, combined with the
    # context-aware projection from inputs_embeds.
    hidden_size_per_layer_input: int = 0
    # Gemma4 E-model vocabulary for the runtime-side token-identity PLE table.
    # Raw checkpoint rows are converted into the runtime pool.
    vocab_size_per_layer_input: int = 0
    # Gemma4: number of KV-shared layers (counted from the last layer backward).
    # Layers [num_hidden_layers - num_kv_shared_layers, num_hidden_layers) are
    # "KV-shared" — in HF they reuse KV states from the last non-shared layer
    # of the same attention type. Our export gives them independent K/V, but
    # they may have a 2× wider MLP (use_double_wide_mlp).
    num_kv_shared_layers: int = 0
    # When True, KV-shared layers use 2× intermediate_size for their MLP.
    use_double_wide_mlp: bool = False
    # Gemma4 26B-A4B MoE: when True, each decoder layer has a routed MoE
    # block in addition to the dense MLP (parallel experts + router).
    enable_moe_block: bool = False
    # ------------------------------------------ tensor parallel
    # ``mapping`` is the single source of truth for parallel placement.
    # tp_size>1 returns a per-rank ONNX graph with col/row-parallel projections.
    mapping: Mapping = field(default_factory=Mapping)

    def __post_init__(self) -> None:
        # For standalone models, num_kv_shared_layers < num_hidden_layers is
        # required so that at least one non-shared donor layer exists.
        # Gemma4 assistant models (shares_target_kv=True) share KV from the
        # *target* model, so num_kv_shared_layers == num_hidden_layers is valid
        # — the donor-index logic is skipped at runtime for those.
        if (self.num_kv_shared_layers > 0
                and self.num_kv_shared_layers >= self.num_hidden_layers
                and not self.shares_target_kv):
            raise ValueError(
                f"num_kv_shared_layers ({self.num_kv_shared_layers}) must be "
                f"less than num_hidden_layers ({self.num_hidden_layers})")

        # Resolve and validate the model family's attention scale.
        self.default_attention_scale = _get_attention_scaling(
            {}, self.head_dim, self.default_attention_scale)
        scale_config = ({} if self.attention_scaling is None else {
            "attention_scaling": self.attention_scaling
        })
        self.attention_scaling = _get_attention_scaling(
            scale_config, self.head_dim, self.default_attention_scale)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def tp_size(self) -> int:
        return self.mapping.tp_size

    @property
    def tp_rank(self) -> int:
        return self.mapping.tp_rank

    @property
    def is_eagle4_draft(self) -> bool:
        """An EAGLE-4 drafter: an EAGLE draft with ``fc_norm``, ``norm_output`` or a low-rank candidate head."""
        return self.is_eagle3_draft and (self.eagle_fc_norm
                                         or self.eagle_norm_output
                                         or self.draft_candidate_count > 0)

    @property
    def is_eagle3_draft(self) -> bool:
        return self.is_eagle3_draft_flag or (
            self.draft_vocab_size is not None and not self.is_dflash_draft
            and not self.dflash_base and not self.is_jetspec_draft
            and not self.jetspec_base and not self.is_dspark_draft
            and not self.dspark_base)

    @property
    def is_mtp_draft(self) -> bool:
        """True for a derived MTP draft config built from a base checkpoint."""
        if self.mtp_base or self.is_eagle3_draft or self.is_dflash_draft \
                or self.is_jetspec_draft or self.is_dspark_draft:
            return False
        return self.is_mtp_draft_flag

    @property
    def is_gemma4_mtp_draft(self) -> bool:
        """True for a paired Gemma4 assistant draft checkpoint."""
        return self.gemma4_mtp_draft

    @property
    def is_diffusion_gemma(self) -> bool:
        return self.diffusion is not None or _is_diffusion_gemma_model_type(
            self.model_type)

    @property
    def is_dflash_draft(self) -> bool:
        return self.is_dflash_draft_flag

    @property
    def is_domino_draft(self) -> bool:
        return (self.is_dflash_draft_flag
                and self.dflash_projector_type == "domino")

    @property
    def domino_base(self) -> bool:
        return self.dflash_base and self.dflash_projector_type == "domino"

    @property
    def is_jetspec_draft(self) -> bool:
        return self.is_jetspec_draft_flag

    @property
    def is_dspark_draft(self) -> bool:
        return self.is_dspark_draft_flag

    @property
    def ple_enabled(self) -> bool:
        """True when Gemma4 per-layer embeddings are enabled."""
        return self.hidden_size_per_layer_input > 0

    @property
    def ngram_layers(self) -> List[int]:
        """Zero-indexed decoder layers carrying a Qwen4-Exp n-gram injection.

        Unrelated to :attr:`ple_enabled`, which is Gemma4's per-layer embeddings; the two features
        share only a name.
        """
        return sorted(index - 1 for index in self.ple_layer_ids)

    @property
    def ngram_heads(self) -> int:
        """Row IDs looked up per token: one per hash head, over all n-gram orders but unigram."""
        return (self.ngram_size - 1) * self.heads_per_ngram

    @property
    def eagle3_target_hidden_size(self) -> int:
        return self.target_hidden_size or self.hidden_size

    @property
    def eagle3_num_target_layers(self) -> int:
        return len(self.eagle3_target_layer_ids) or 3

    @property
    def is_hybrid(self) -> bool:
        return self.mamba_cfg is not None or self.gdn_cfg is not None

    @property
    def is_nemotron_h(self) -> bool:
        return (self.model_type or "").lower().startswith("nemotron_h")

    @property
    def num_attn_layers(self) -> int:
        """Total attention layers (includes Gemma4 sliding/full variants).

        Note: layers may have heterogeneous head dims — use per-layer configs
        (kv_layer_configs) for allocation, not this count alone.
        """
        return sum(1 for t in self.layer_types
                   if t in (LAYER_ATTN, "sliding_attention", "full_attention"))

    @property
    def kv_pool_head_dim(self) -> int:
        """Paged KV pool row width of an attention layer.

        A checkpoint with an indexer (``indexer_head_dim > 0``) runs every
        attention layer as Qwen Sparse Attention, and QsaAttentionPlugin
        persists the indexer state in a tail of ``indexer_head_dim`` columns
        after the ``head_dim`` K/V columns; without an indexer the row is
        ``head_dim`` (Gemma4's global layers are sized by their own per-layer
        hook). The exported ``kv_layer_configs`` and a model's
        ``past_key_values`` export spec must both size the pool with this,
        while the top-level ``head_dim`` (which sizes the rotary binding)
        stays untouched.
        """
        return self.head_dim + self.indexer_head_dim

    @property
    def num_mamba_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == LAYER_MAMBA)

    @property
    def num_gdn_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == LAYER_GDN)

    @property
    def num_mlp_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == LAYER_MLP)

    @property
    def num_moe_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == LAYER_MOE)

    @property
    def use_dual_rope(self) -> bool:
        return (self.sliding_rope_config is not None
                and self.full_rope_config is not None)

    @property
    def compute_dtype(self) -> "torch.dtype":  # noqa: F821
        import torch
        _MAP = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return _MAP.get(self.torch_dtype, torch.bfloat16)

    def for_rank(self, rank: int, world: int) -> "ModelConfig":
        """Return a per-rank copy of this config for TP.

        Divides head and intermediate sizes by *world* so each rank's model
        carries per-rank shapes.

        Usage:
            cfg = ModelConfig.from_pretrained(
                path, lambda head_dim: 1.0 / (float(head_dim)**0.5)
            ).for_rank(rank, world)
            model = CausalLM(cfg)
            load_weights(model, path, mapping=cfg.mapping)
        """
        import copy
        if world == 1:
            return self
        parallel_dimensions = [
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("intermediate_size", self.intermediate_size),
        ]
        if self.gdn_cfg is not None:
            parallel_dimensions.extend((
                ("gdn_cfg.num_key_heads", self.gdn_cfg.num_key_heads),
                ("gdn_cfg.num_value_heads", self.gdn_cfg.num_value_heads),
            ))
        for name, v in parallel_dimensions:
            if v % world:
                raise ValueError(
                    f"TP world={world}: {name}={v} is not divisible by {world}"
                )
        c = copy.deepcopy(self)
        c.mapping = Mapping(world_size=world,
                            rank=rank,
                            tp_size=world,
                            tp_rank=rank)
        c.num_attention_heads //= world
        c.num_key_value_heads //= world
        c.intermediate_size //= world
        if c.gdn_cfg is not None:
            c.gdn_cfg.num_key_heads //= world
            c.gdn_cfg.num_value_heads //= world
        return c

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
            cls, model_dir: str,
            default_attention_scale: Callable[[int], float]) -> "ModelConfig":
        """Load a ModelConfig from a checkpoint directory.

        Loads architecture hyper-parameters via ``AutoConfig`` (see
        :func:`checkpoint_utils.load_checkpoint_config_dicts`) and then
        either ``hf_quant_config.json`` or the embedded ``quantization_config``
        block to determine
        the quantisation scheme.

        ``default_attention_scale`` is a required model-family callable
        accepting ``head_dim``. ``has_qk_norm`` is auto-detected by scanning
        the safetensors key index for ``.q_norm.weight`` entries; the HunYuan
        ``.query_layernorm.weight`` spelling is honored only for model types
        whose loader remaps it (see :func:`_detect_has_qk_norm`).
        """
        root, llm_dict = load_checkpoint_config_dicts(model_dir)
        if llm_dict.get("num_hidden_layers") is None:
            layer_schedule = (llm_dict.get("layers_block_type")
                              or llm_dict.get("layer_types"))
            if isinstance(layer_schedule, (list, tuple)) and layer_schedule:
                llm_dict = dict(llm_dict)
                llm_dict["num_hidden_layers"] = len(layer_schedule)

        root_model_type = root.get("model_type", "")
        model_type = llm_dict.get("model_type", "llama")
        is_diffusion_gemma = _is_diffusion_gemma_config(root, llm_dict)
        is_gemma4_assistant = _is_gemma4_assistant_model_type(root_model_type)
        if is_gemma4_assistant:
            model_type = "gemma4_assistant"
        elif is_diffusion_gemma:
            model_type = "diffusion_gemma"
        hidden_size = llm_dict["hidden_size"]
        num_attn_heads = llm_dict["num_attention_heads"]
        _check_num_attention_heads(num_attn_heads)
        head_dim = llm_dict.get("head_dim", hidden_size // num_attn_heads)
        global_head_dim = int(llm_dict.get("global_head_dim", 0) or 0)
        num_global_kv_heads = int(
            llm_dict.get("num_global_key_value_heads", 0) or 0)
        indexer_head_dim = int(llm_dict.get("indexer_head_dim", 0) or 0)

        # Omni roots keep one ignore list for thinker + talker; scope it to the
        # sub-model this config represents (talker/CP go through their own
        # staged sub-config instead).
        submodel_prefix = ""
        if root.get("thinker_config") is not None and llm_dict is not root:
            submodel_prefix = "thinker."
        quant_dict = llm_dict
        if ("quantization_config" not in quant_dict
                and root.get("quantization_config") is not None):
            quant_dict = dict(llm_dict)
            quant_dict["quantization_config"] = root["quantization_config"]
        quant = _parse_quant(model_dir, quant_dict, submodel_prefix)
        raw_layer_types = _parse_raw_layer_types(llm_dict)
        layer_types = _parse_layer_types(llm_dict)
        block_configs = _parse_puzzle_block_configs(llm_dict, layer_types)
        attention_layer_types = _parse_attention_layer_types(
            llm_dict, llm_dict["num_hidden_layers"], model_type)
        num_kv_heads = llm_dict.get("num_key_value_heads", num_attn_heads)
        full_attention_only = (_is_gemma4_model_type(model_type)
                               and attention_layer_types and all(
                                   layer_type == "full_attention"
                                   for layer_type in attention_layer_types))
        if full_attention_only:
            if global_head_dim:
                head_dim = global_head_dim
            if num_global_kv_heads:
                num_kv_heads = num_global_kv_heads
        dual_rope_configs = _get_dual_rope_configs(llm_dict)
        mamba_cfg = _parse_mamba_cfg(llm_dict,
                                     layer_types,
                                     model_dir=model_dir)
        gdn_cfg = _parse_gdn_cfg(llm_dict, layer_types)
        has_qk_norm = _detect_has_qk_norm(model_dir, model_type)
        has_value_norm = _get_has_value_norm(llm_dict, model_type)
        default_attention_scale_value = float(
            default_attention_scale(head_dim))
        attention_scaling = _get_attention_scaling(
            llm_dict, head_dim, default_attention_scale_value)
        embedding_scale = _get_embedding_scale(llm_dict, model_type,
                                               hidden_size)

        if _is_muse_glimmer_model_type(model_type):
            qk_scale_factor = llm_dict.get("qk_scale_factor")
            if qk_scale_factor is not None:
                attention_scaling = float(qk_scale_factor) / math.sqrt(
                    head_dim)
            # Full-attention layers use NoPE; sliding layers use regular RoPE.
            if not dual_rope_configs:
                _mp = int(llm_dict.get("max_position_embeddings", 4096))
                _sliding = {
                    "rope_theta": _get_rope_theta(llm_dict),
                    "rope_scaling": None,
                    "partial_rotary_factor": 1.0,
                    "max_position_embeddings": _mp,
                }
                _full = dict(_sliding)
                _full["rope_scaling"] = {"rope_type": "nope", "type": "nope"}
                dual_rope_configs = {
                    "sliding_rope_config": _sliding,
                    "full_rope_config": _full,
                }

        generation_config_path = os.path.join(model_dir,
                                              "generation_config.json")
        generation_config: Dict[str, Any] = {}
        if os.path.isfile(generation_config_path):
            with open(generation_config_path) as f:
                generation_config = json.load(f)
        diffusion_source_config = dict(root)
        diffusion_source_config.update(llm_dict)
        diffusion_config = (DiffusionConfig.from_hf(diffusion_source_config,
                                                    generation_config)
                            if is_diffusion_gemma else None)
        self_conditioning_size = 0

        # MTP config
        mtp_num_hidden_layers = llm_dict.get("mtp_num_hidden_layers")
        if mtp_num_hidden_layers is not None:
            mtp_num_hidden_layers = int(mtp_num_hidden_layers)
        mtp_hybrid_override_pattern = llm_dict.get(
            "mtp_hybrid_override_pattern")
        mtp_layer_types = _parse_mtp_layer_types(llm_dict)
        num_nextn_predict_layers = int(
            llm_dict.get("num_nextn_predict_layers", 0) or 0)
        if (mtp_num_hidden_layers is None and num_nextn_predict_layers > 0
                and mtp_layer_types):
            mtp_num_hidden_layers = len(mtp_layer_types)
        elif (mtp_num_hidden_layers is None and num_nextn_predict_layers > 0
              and model_type in _MTP_CONFIG_MODEL_TYPES):
            # Dense Qwen3 MTP heads (e.g. AngelSlim/Qwen3-8B-MTP) declare only
            # the DeepSeek-style prediction-module count.
            mtp_num_hidden_layers = num_nextn_predict_layers
        mtp_use_dedicated_embeddings = bool(
            llm_dict.get("mtp_use_dedicated_embeddings", False))
        _validate_mtp_constraints(
            model_type=model_type,
            mtp_num_hidden_layers=mtp_num_hidden_layers,
            mtp_use_dedicated_embeddings=mtp_use_dedicated_embeddings,
        )

        # EAGLE3 draft model fields
        speculator_type = llm_dict.get("speculators_model_type", "")
        if speculator_type == "peagle":
            raise NotImplementedError(
                "P-EAGLE uses a parallel drafting contract that is not supported "
                "by the Edge-LLM EAGLE3 runtime.")
        architectures = llm_dict.get("architectures", []) or []
        is_eagle3_draft_flag = any(
            "Eagle3" in str(arch) for arch in architectures) or (
                "ttt_length" in llm_dict) or speculator_type == "eagle3"
        has_eagle4_features = bool(_eagle_draft_feature_keys(llm_dict))
        if has_eagle4_features:
            # Drafts with fc_norm, norm_output or a candidate head also mark
            # themselves with an Eagle4 architecture or TorchSpec's
            # eagle_config.use_aux_hidden_state.
            is_eagle3_draft_flag = is_eagle3_draft_flag or any(
                "Eagle4" in str(arch) for arch in architectures) or bool(
                    (llm_dict.get("eagle_config")
                     or {}).get("use_aux_hidden_state"))
        draft_vocab_size = llm_dict.get("draft_vocab_size", None)
        target_hidden_size = llm_dict.get("target_hidden_size", None)
        eagle3_target_layer_ids = list(
            llm_dict.get("target_layer_ids", [])
            or []) if is_eagle3_draft_flag else []
        eagle_features = _EagleDraftFeatures()
        if is_eagle3_draft_flag and has_eagle4_features:
            eagle3_target_layer_ids = _parse_eagle_target_layer_ids(llm_dict)
            eagle_features = _parse_eagle_draft_features(
                llm_dict, eagle3_target_layer_ids)
        if is_eagle3_draft_flag:
            draft_vocab_size = draft_vocab_size or llm_dict.get("vocab_size")
            target_hidden_size = target_hidden_size or hidden_size
        # Sliding window: active when use_sliding_window=True, or when
        # layer_types contains "sliding_attention" (Gemma4 convention).
        use_sw = llm_dict.get("use_sliding_window", False) or any(
            layer_type == "sliding_attention"
            for layer_type in raw_layer_types)
        sw_raw = llm_dict.get("sliding_window") if use_sw else None
        sliding_window_size = int(sw_raw) if sw_raw is not None else -1
        skip_softmax_prefill_enabled = bool(
            llm_dict.get("skip_softmax_prefill_enabled", False))
        skip_softmax_decode_enabled = bool(
            llm_dict.get("skip_softmax_decode_enabled", False))
        use_vision_bidirectional_attention = bool(
            model_type in ("gemma4_unified", "gemma4_unified_text")
            and llm_dict.get("use_bidirectional_attention") == "vision")

        jetspec_config = (llm_dict.get("jetspec_config")
                          or llm_dict.get("dflash_config") or {})
        jetspec_target_layer_ids = list(
            jetspec_config.get("target_layer_ids",
                               llm_dict.get("target_layer_ids", [])) or [])
        jetspec_block_size = int(
            jetspec_config.get("block_size", llm_dict.get("block_size", 16)))
        jetspec_mask_token_id = int(
            jetspec_config.get("mask_token_id",
                               llm_dict.get("mask_token_id", 151669)))
        jetspec_causal_head = bool(
            jetspec_config.get("causal_head",
                               llm_dict.get("causal_head", True)))

        # Sparse MoE fields.  HF uses "num_local_experts" as the internal key
        # and maps "num_experts" → "num_local_experts" via attribute_map.
        num_experts = int(
            llm_dict.get("num_experts", llm_dict.get("num_local_experts", 0))
            or 0)
        # Nemotron-H declares routed experts as ``n_routed_experts`` rather
        # than ``num_experts`` / ``num_local_experts``; treat either as MoE so
        # the sizes below are parsed instead of defaulting to 0.
        n_routed_experts = int(llm_dict.get("n_routed_experts", 0) or 0)
        if num_experts > 0 or n_routed_experts > 0:
            num_experts_per_tok = int(
                llm_dict.get("num_experts_per_tok",
                             llm_dict.get("top_k_experts", 0)) or 0)
            moe_intermediate_size = int(
                llm_dict.get("moe_intermediate_size", 0) or 0)
        else:
            num_experts_per_tok = 0
            moe_intermediate_size = 0
        # HF Qwen3-Omni MoE Talker uses the un-prefixed name; HF NemotronH /
        # other MoE families use ``moe_shared_expert_intermediate_size``.
        moe_shared_expert_intermediate_size = int(
            llm_dict.get("moe_shared_expert_intermediate_size",
                         llm_dict.get("shared_expert_intermediate_size", 0))
            or 0)
        routed_scaling_factor = float(
            llm_dict.get("routed_scaling_factor", 1.0))
        n_group = int(llm_dict.get("n_group", 1))
        topk_group = int(llm_dict.get("topk_group", 1))
        decoder_sparse_step = int(llm_dict.get("decoder_sparse_step", 1))
        mlp_only_layers = list(llm_dict.get("mlp_only_layers") or [])
        norm_topk_prob = bool(llm_dict.get("norm_topk_prob", True))
        moe_latent_size = llm_dict.get("moe_latent_size", None)
        if moe_latent_size is not None:
            moe_latent_size = int(moe_latent_size)

        intermediate_size = int(
            llm_dict.get("intermediate_size")
            or llm_dict.get("shared_expert_intermediate_size")
            or llm_dict.get("moe_shared_expert_intermediate_size")
            or llm_dict.get("moe_intermediate_size", 0))
        if self_conditioning_size <= 0:
            self_conditioning_size = int(
                llm_dict.get(
                    "self_conditioning_size",
                    root.get("self_conditioning_size", intermediate_size))
                or intermediate_size)

        config = cls(
            model_type=model_type,
            hidden_size=hidden_size,
            num_hidden_layers=llm_dict["num_hidden_layers"],
            num_attention_heads=num_attn_heads,
            num_key_value_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            global_head_dim=global_head_dim,
            num_global_key_value_heads=num_global_kv_heads,
            indexer_head_dim=indexer_head_dim,
            rms_norm_eps=_get_rms_norm_eps(llm_dict, model_type),
            vocab_size=llm_dict["vocab_size"],
            rope_theta=_get_rope_theta(llm_dict),
            max_position_embeddings=llm_dict.get("max_position_embeddings",
                                                 4096),
            default_attention_scale=default_attention_scale_value,
            rope_scaling=_select_rope_scaling(llm_dict),
            original_max_position_embeddings=llm_dict.get(
                "original_max_position_embeddings", None),
            partial_rotary_factor=_get_partial_rotary_factor(llm_dict),
            hidden_activation=llm_dict.get("hidden_activation",
                                           llm_dict.get("hidden_act", "silu")),
            num_code_groups=int(llm_dict.get("num_code_groups", 0) or 0),
            sliding_rope_config=dual_rope_configs.get("sliding_rope_config"),
            full_rope_config=dual_rope_configs.get("full_rope_config"),
            has_qk_norm=has_qk_norm,
            qk_norm_post_rope=(model_type in _QK_NORM_POST_ROPE_MODEL_TYPES),
            has_value_norm=has_value_norm,
            attention_bias=bool(llm_dict.get("attention_bias", False)),
            attention_scaling=attention_scaling,
            attention_k_eq_v=(True if is_diffusion_gemma else bool(
                llm_dict.get("attention_k_eq_v", False))),
            encoder_layer_scalars=list(
                llm_dict.get("encoder_layer_scalars",
                             root.get("encoder_layer_scalars", [])) or []),
            decoder_layer_scalars=list(
                llm_dict.get("decoder_layer_scalars",
                             root.get("decoder_layer_scalars", [])) or []),
            self_conditioning_size=self_conditioning_size,
            diffusion=diffusion_config,
            embedding_scale=embedding_scale,
            final_logit_softcapping=llm_dict.get("final_logit_softcapping",
                                                 None),
            output_multiplier=float(llm_dict.get("output_multiplier", 1.0)),
            post_norm_eps=(float(llm_dict["post_norm_eps"])
                           if llm_dict.get("post_norm_eps") is not None else
                           None),
            torch_dtype=llm_dict.get("torch_dtype",
                                     llm_dict.get("dtype", "bfloat16")),
            tie_word_embeddings=llm_dict.get("tie_word_embeddings", False),
            sliding_window_size=sliding_window_size,
            skip_softmax_prefill_enabled=skip_softmax_prefill_enabled,
            skip_softmax_decode_enabled=skip_softmax_decode_enabled,
            use_vision_bidirectional_attention=
            use_vision_bidirectional_attention,
            layer_types=layer_types,
            block_configs=block_configs,
            attention_layer_types=attention_layer_types,
            quant=quant,
            mamba_cfg=mamba_cfg,
            gdn_cfg=gdn_cfg,
            attn_output_gate=bool(llm_dict.get("attn_output_gate", False)),
            hc_count=int(llm_dict.get("hc_count", 0)),
            hc_lowrank=int(llm_dict.get("hc_lowrank", 0)),
            indexer_n_heads=int(llm_dict.get("indexer_n_heads") or 0),
            indexer_kv_heads=int(llm_dict.get("indexer_kv_heads") or 0),
            indexer_budget=int(llm_dict.get("indexer_budget") or 0),
            indexer_compress_ratio=int(
                llm_dict.get("indexer_compress_ratio") or 0),
            ple_layer_ids=[
                int(i) for i in (llm_dict.get("ple_layer_ids") or [])
            ],
            ple_embed_dim=int(llm_dict.get("ple_embed_dim", 0)),
            ple_conv_kernel_size=int(llm_dict.get("ple_conv_kernel_size", 0)),
            ngram_size=int(llm_dict.get("ngram_size", 0)),
            heads_per_ngram=int(llm_dict.get("heads_per_ngram", 0)),
            split_ngram_parts=int(llm_dict.get("split_ngram_parts") or 0),
            mtp_num_hidden_layers=mtp_num_hidden_layers,
            mtp_use_dedicated_embeddings=mtp_use_dedicated_embeddings,
            mtp_hybrid_override_pattern=mtp_hybrid_override_pattern,
            mtp_layer_types=mtp_layer_types,
            mtp_base=bool(llm_dict.get("mtp_base", False)),
            root_model_type=root_model_type,
            raw_layer_types=raw_layer_types,
            rope_parameters=llm_dict.get("rope_parameters", None),
            backbone_hidden_size=int(
                llm_dict.get("backbone_hidden_size", 0) or 0),
            assistant_hidden_size=(hidden_size if is_gemma4_assistant else 0),
            shares_target_kv=is_gemma4_assistant,
            has_own_kv_cache=not is_gemma4_assistant,
            constant_draft_positions=is_gemma4_assistant,
            returns_feedback_hidden=is_gemma4_assistant,
            use_ordered_embeddings=bool(
                llm_dict.get("use_ordered_embeddings", False)),
            num_centroids=int(llm_dict.get("num_centroids", 0) or 0),
            centroid_intermediate_top_k=int(
                llm_dict.get("centroid_intermediate_top_k", 0) or 0),
            dflash_base=bool(llm_dict.get("dflash_base", False)),
            jetspec_base=bool(llm_dict.get("jetspec_base", False)),
            jetspec_target_layer_ids=jetspec_target_layer_ids,
            jetspec_block_size=jetspec_block_size,
            jetspec_mask_token_id=jetspec_mask_token_id,
            jetspec_causal_head=jetspec_causal_head,
            dflash_target_layer_ids=list(
                (llm_dict.get("dflash_config", {})
                 or {}).get("target_layer_ids")
                or llm_dict.get("eagle_aux_hidden_state_layer_ids") or []),
            dspark_base=bool(llm_dict.get("dspark_base", False)),
            num_deepstack_features=_parse_num_deepstack_features(
                llm_dict, model_type, root_config=root),
            accept_hidden_layer=_parse_accept_hidden_layer(llm_dict,
                                                           root_config=root),
            draft_vocab_size=draft_vocab_size,
            target_hidden_size=target_hidden_size,
            is_eagle3_draft_flag=is_eagle3_draft_flag,
            eagle3_target_layer_ids=eagle3_target_layer_ids,
            eagle3_norm_before_residual=bool(
                llm_dict.get("norm_before_residual", False)),
            eagle3_norm_before_fc=bool(llm_dict.get("norm_before_fc", False)),
            eagle_fc_norm=eagle_features.fc_norm,
            eagle_norm_output=eagle_features.norm_output,
            draft_candidate_rank=eagle_features.candidate_rank,
            draft_candidate_count=eagle_features.candidate_count,
            num_experts=num_experts,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            moe_shared_expert_intermediate_size=
            moe_shared_expert_intermediate_size,
            moe_latent_size=moe_latent_size,
            routed_scaling_factor=routed_scaling_factor,
            n_group=n_group,
            topk_group=topk_group,
            decoder_sparse_step=decoder_sparse_step,
            mlp_only_layers=mlp_only_layers,
            norm_topk_prob=norm_topk_prob,
            hidden_size_per_layer_input=int(
                llm_dict.get("hidden_size_per_layer_input", 0) or 0),
            vocab_size_per_layer_input=int(
                llm_dict.get("vocab_size_per_layer_input", 0) or 0),
            num_kv_shared_layers=int(
                llm_dict.get("num_kv_shared_layers", 0) or 0),
            use_double_wide_mlp=bool(llm_dict.get("use_double_wide_mlp",
                                                  False)),
            enable_moe_block=bool(
                llm_dict.get(
                    "enable_moe_block", model_type in (
                        "gemma4",
                        "gemma4_text",
                        "gemma4_unified",
                        "gemma4_unified_text",
                        "diffusion_gemma",
                    ) and num_experts > 0 and moe_intermediate_size > 0)),
        )
        if has_eagle4_features and not is_eagle3_draft_flag \
                and config.is_eagle3_draft:
            raise ValueError(
                "This EAGLE draft config sets "
                f"{_describe_eagle_draft_features(llm_dict)} but no EAGLE "
                "draft marker (an Eagle3 or Eagle4 architecture, ttt_length or "
                "eagle_config.use_aux_hidden_state), so it would export as a "
                "dense EAGLE3 draft without them. Add the draft's Eagle4 "
                "architecture to its config.json.")
        return config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# When HF omits ``deepstack_visual_indexes`` / ``num_deepstack_features``,
# these model_types are known to expect three visual deepstack injections
# at runtime (Thinker side only). Listed explicitly — substring matching
# (``"qwen3_omni" in "qwen3_omni_moe_talker"``) would otherwise mis-classify
# Talker / CodePredictor configs as deepstack producers and bake 3 dangling
# input ports into their engines.
_DEEPSTACK_MODEL_TYPES = frozenset({
    # HF root configs that wrap a deepstack-emitting visual encoder
    "qwen3_vl",
    "qwen3_omni",
    "qwen3_omni_moe",
    # Standalone Thinker text-LLM configs (after quant export); the Thinker
    # still consumes deepstack inputs from the separately exported visual
    # encoder at runtime.
    "qwen3_vl_text",
    "qwen3_omni_text",
    "qwen3_omni_moe_text",
})

# Model types whose checkpoints may carry a Qwen3-Next-layout ``mtp.*`` head.
# Dense ``qwen3`` is served by the same draft class with plain decoder layers
# (validated with AngelSlim/Qwen3-8B-MTP); add a family only with a validated head.
_MTP_CONFIG_MODEL_TYPES = frozenset({
    "qwen3",
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen3_omni_next_text",
    "qwen3_omni_next_text_moe",
    "qwen4_exp_text",
})

# Qwen3.5-family model types (checkpoint and text-config) whose hybrid base
# exports the EAGLE verify ABI. Qwen3-Omni-Next reuses the Qwen3.5 classes but
# is not one of them.
_QWEN35_EAGLE_BASE_MODEL_TYPES = frozenset({
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
})


def make_mtp_draft_config(base_config: ModelConfig) -> ModelConfig:
    """Derive the currently supported MTP draft config from a base config."""
    _validate_mtp_constraints(
        model_type=base_config.model_type,
        mtp_num_hidden_layers=base_config.mtp_num_hidden_layers,
        mtp_use_dedicated_embeddings=base_config.mtp_use_dedicated_embeddings,
    )
    mtp_num_hidden_layers = base_config.mtp_num_hidden_layers
    if mtp_num_hidden_layers is None:
        raise ValueError(
            "MTP draft config requires mtp_num_hidden_layers in the base config."
        )

    # Nemotron-H: Exclude all draft ``layers.*`` modules from quantization;
    # the untouched lm_head keeps the base quant type.
    if (base_config.model_type or "").lower().startswith("nemotron_h"):
        draft_layer_types = list(base_config.mtp_layer_types)
        if len(draft_layer_types) != mtp_num_hidden_layers:
            declared = (base_config.mtp_hybrid_override_pattern
                        or base_config.mtp_layer_types)
            raise ValueError(
                f"MTP draft stack {declared!r} yields "
                f"{len(draft_layer_types)} layers != mtp_num_hidden_layers "
                f"{mtp_num_hidden_layers}")
        draft_quant = replace(
            base_config.quant,
            excluded=list(base_config.quant.excluded) + ["layers.*"],
        )
        return replace(
            base_config,
            num_hidden_layers=mtp_num_hidden_layers,
            layer_types=draft_layer_types,
            mamba_cfg=None,
            gdn_cfg=None,
            mtp_base=False,
            is_mtp_draft_flag=True,
            quant=draft_quant,
            tie_word_embeddings=False,
        )

    # The draft is quantized iff its FFN compute weights are quantized
    # (routed experts for MoE, dense MLP otherwise) — other modules may be
    # excluded in any recipe. Excluded entries are exact names or globs.
    ffn_probe = ("mtp.layers.0.mlp.experts.0.gate_proj" if
                 base_config.num_experts > 0 else "mtp.layers.0.mlp.gate_proj")
    mtp_is_quantized = not any(
        fnmatch.fnmatch(ffn_probe, e) for e in base_config.quant.excluded)

    if mtp_is_quantized:
        # MTP draft modules are independently quantized.  Strip base-model
        # layer_overrides that use module paths absent from the draft
        # (e.g. "model.layers.0.linear_attn.in_proj_qkv") — they would
        # cause spurious FP16 fallback for draft-specific layers like "fc".
        # Keep entries whose keys also exist in the draft module namespace
        # (e.g. "lm_head") so that lm_head_quantization is honoured.
        _DRAFT_MODULE_PREFIXES = ("lm_head", "fc", "layers.", "norm")

        def _draft_entries(entries: dict) -> dict:
            # ``mtp.``-prefixed entries name draft modules and win over
            # same-named base entries.
            kept = {
                k: v
                for k, v in entries.items() if any(
                    k == p or k.startswith(p) for p in _DRAFT_MODULE_PREFIXES)
            }
            kept.update({
                k[len("mtp."):]: v
                for k, v in entries.items() if k.startswith("mtp.")
            })
            return kept

        draft_overrides = _draft_entries(base_config.quant.layer_overrides)
        draft_group_sizes = _draft_entries(base_config.quant.layer_group_sizes)
        draft_listed = any(
            k.startswith("mtp.") for k in base_config.quant.layer_overrides)
        # Preserve MTP-specific exclusions (e.g. mtp.lm_head when lm_head
        # is FP16) but drop base-model exclusions irrelevant to the draft.
        draft_excluded = [
            e[len("mtp."):] for e in base_config.quant.excluded
            if e.startswith("mtp.")
        ]
        # A checkpoint that lists draft modules keeps mixed precision
        # (unlisted draft modules are FP16). One that lists none sends
        # unlisted modules (fc, q_proj, etc.) to the dominant quant type;
        # explicit overrides (lm_head→fp8) still apply.
        draft_quant = replace(
            base_config.quant,
            excluded=draft_excluded,
            layer_overrides=draft_overrides,
            layer_group_sizes=draft_group_sizes,
            is_mixed_precision=(base_config.quant.is_mixed_precision
                                and draft_listed))
    else:
        # MTP draft lm_head is borrowed from the base model and may itself be quantized.
        lm_head_overrides = {
            k: v
            for k, v in base_config.quant.layer_overrides.items()
            if k == "lm_head" or k.startswith("lm_head.")
        }
        lm_head_group_sizes = {
            k: v
            for k, v in base_config.quant.layer_group_sizes.items()
            if k == "lm_head" or k.startswith("lm_head.")
        }
        draft_quant = QuantConfig(layer_overrides=lm_head_overrides,
                                  layer_group_sizes=lm_head_group_sizes)

    return replace(
        base_config,
        num_hidden_layers=mtp_num_hidden_layers,
        layer_types=[LAYER_ATTN] * mtp_num_hidden_layers,
        gdn_cfg=None,
        ple_layer_ids=[],
        mtp_base=False,
        is_mtp_draft_flag=True,
        quant=draft_quant,
        tie_word_embeddings=False,
    )


def make_dspark_draft_config(
        draft_dir: str,
        default_attention_scale: Callable[[int], float]) -> ModelConfig:
    """Build a DSpark draft ModelConfig from a DeepSpec DSpark checkpoint."""
    _, llm_dict = load_checkpoint_config_dicts(draft_dir)
    dspark_config = (llm_dict.get("dspark_config")
                     or llm_dict.get("dflash_config", {}) or {})

    quant = _parse_quant(draft_dir, llm_dict)
    target_layer_ids = list(
        dspark_config.get("target_layer_ids",
                          llm_dict.get("target_layer_ids", [])))
    if not target_layer_ids:
        raise ValueError(
            "DSpark draft config requires target_layer_ids in config.json.")

    model_type = llm_dict.get("model_type", "qwen3")
    raw_layer_types = _parse_raw_layer_types(llm_dict)
    layer_types = _parse_layer_types(llm_dict)
    attention_layer_types = _parse_attention_layer_types(
        llm_dict, llm_dict["num_hidden_layers"], model_type)
    if (not attention_layer_types and raw_layer_types
            and all(layer_type in _VALID_ATTENTION_LAYER_TYPES
                    for layer_type in raw_layer_types)):
        attention_layer_types = list(raw_layer_types)
    full_attention_only = (_is_gemma4_model_type(model_type)
                           and attention_layer_types
                           and all(layer_type == "full_attention"
                                   for layer_type in attention_layer_types))
    head_dim = llm_dict.get(
        "global_head_dim" if full_attention_only else "head_dim",
        llm_dict.get(
            "head_dim",
            llm_dict["hidden_size"] // llm_dict["num_attention_heads"]))
    global_head_dim = int(llm_dict.get("global_head_dim", 0) or 0)
    num_global_kv_heads = int(
        llm_dict.get("num_global_key_value_heads", 0) or 0)
    num_kv_heads = (
        num_global_kv_heads if full_attention_only and num_global_kv_heads else
        llm_dict.get("num_key_value_heads", llm_dict["num_attention_heads"]))
    dual_rope_configs = _get_dual_rope_configs(llm_dict)
    use_sw = llm_dict.get("use_sliding_window", False) or any(
        layer_type == "sliding_attention" for layer_type in raw_layer_types)
    sw_raw = llm_dict.get("sliding_window") if use_sw else None
    sliding_window_size = int(sw_raw) if sw_raw is not None else -1
    default_attention_scale_value = float(default_attention_scale(head_dim))
    default_mask_token_id = 4 if _is_gemma4_model_type(model_type) else 151669

    draft_vocab_size = llm_dict.get("draft_vocab_size")
    draft_vocab_size = int(llm_dict["vocab_size"] if draft_vocab_size is
                           None else draft_vocab_size)
    return ModelConfig(
        model_type=model_type,
        hidden_size=llm_dict["hidden_size"],
        num_hidden_layers=llm_dict["num_hidden_layers"],
        num_attention_heads=llm_dict["num_attention_heads"],
        num_key_value_heads=num_kv_heads,
        intermediate_size=llm_dict["intermediate_size"],
        head_dim=head_dim,
        global_head_dim=global_head_dim,
        num_global_key_value_heads=num_global_kv_heads,
        rms_norm_eps=llm_dict.get("rms_norm_eps", 1e-6),
        vocab_size=llm_dict["vocab_size"],
        rope_theta=_get_rope_theta(llm_dict),
        max_position_embeddings=llm_dict.get("max_position_embeddings", 4096),
        default_attention_scale=default_attention_scale_value,
        rope_scaling=_select_rope_scaling(llm_dict),
        partial_rotary_factor=_get_partial_rotary_factor(llm_dict),
        hidden_activation=llm_dict.get("hidden_activation",
                                       llm_dict.get("hidden_act", "silu")),
        sliding_rope_config=dual_rope_configs.get("sliding_rope_config"),
        full_rope_config=dual_rope_configs.get("full_rope_config"),
        has_qk_norm=True,
        has_value_norm=_get_has_value_norm(llm_dict, model_type),
        attention_bias=bool(llm_dict.get("attention_bias", False)),
        attention_scaling=_get_attention_scaling(
            llm_dict, head_dim, default_attention_scale_value),
        attention_k_eq_v=bool(llm_dict.get("attention_k_eq_v", False)),
        attention_sink_bias=bool(
            dspark_config.get("attention_sink_bias",
                              llm_dict.get("attention_sink_bias", False))),
        final_logit_softcapping=llm_dict.get("final_logit_softcapping", None),
        torch_dtype=llm_dict.get("torch_dtype",
                                 llm_dict.get("dtype", "bfloat16")),
        tie_word_embeddings=False,
        sliding_window_size=sliding_window_size,
        layer_types=layer_types,
        attention_layer_types=attention_layer_types,
        raw_layer_types=raw_layer_types,
        rope_parameters=llm_dict.get("rope_parameters", None),
        draft_vocab_size=draft_vocab_size,
        reduced_vocab_size=(draft_vocab_size if draft_vocab_size
                            < llm_dict["vocab_size"] else None),
        is_dspark_draft_flag=True,
        dspark_causal_proposal=bool(
            dspark_config.get(
                "causal",
                llm_dict.get("dflash_query_causal",
                             (llm_dict.get("dflash_config")
                              or {}).get("causal", False)))),
        dspark_target_layer_ids=target_layer_ids,
        dspark_block_size=int(
            dspark_config.get("block_size", llm_dict.get("block_size", 7))),
        dspark_mask_token_id=int(
            dspark_config.get(
                "mask_token_id",
                llm_dict.get("mask_token_id", default_mask_token_id))),
        dspark_enable_confidence_head=bool(
            dspark_config.get("enable_confidence_head",
                              llm_dict.get("enable_confidence_head", False))),
        dspark_sample_from_anchor=bool(
            dspark_config.get("sample_from_anchor",
                              llm_dict.get("sample_from_anchor", True))),
        dspark_confidence_head_with_markov=bool(
            dspark_config.get(
                "confidence_head_with_markov",
                llm_dict.get("confidence_head_with_markov", False))),
        dspark_markov_head_type=str(
            dspark_config.get("markov_head_type",
                              llm_dict.get("markov_head_type", ""))),
        dspark_markov_rank=int(
            dspark_config.get("markov_rank", llm_dict.get("markov_rank", 0))),
        quant=quant,
    )


def make_dflash_draft_config(
        draft_dir: str,
        default_attention_scale: Callable[[int], float],
        target_vocab_size: Optional[int] = None,
        quantize_w4a16_activations: bool = False) -> ModelConfig:
    """Build a DFlash draft ModelConfig from the draft checkpoint directory.

    ``quantize_w4a16_activations`` runs a declared W4A16 NVFP4 draft as W4A4 for
    the modules that ship an input scale.

    Now quantization-aware: if the draft directory contains
    ``hf_quant_config.json`` (e.g. from DFlash draft NVFP4 quantization),
    the quant config is parsed so ``make_linear`` will produce the correct
    Linear class (``NVFP4Linear`` etc.) during ONNX export.
    """
    root_dict, llm_dict = load_checkpoint_config_dicts(draft_dir)
    dflash_config = llm_dict.get("dflash_config", {}) or {}
    contract = resolve_dflash_contract(root_dict, llm_dict)
    # Domino applies its own label shift inside the correction head.
    shift_label = contract.shift_label and not contract.is_domino

    # Parse quantization config from the draft checkpoint directory.
    # For FP16 draft checkpoints this returns QuantConfig() (no quant).
    quant = _parse_quant(draft_dir, llm_dict)
    if (quantize_w4a16_activations and _DEFAULT_QUANTIZE_ACTIVATIONS
            and quant.quant_type == QUANT_NVFP4_A16):
        # The checkpoint's input scales decide: modules without one fall back
        # to W4A16, whatever the declared algorithm says.
        quant = _weight_only_nvfp4(draft_dir,
                                   replace(quant, quant_type=QUANT_NVFP4), "")
    # A dense fc keeps the FP16 + FP32 projection (the draft model asserts
    # this); a quantized fc runs at the checkpoint's precision, see ``model.py``.
    if "fc" not in quant.excluded:
        quant.excluded = list(quant.excluded) + ["fc"]

    model_type = llm_dict.get("model_type", "qwen3")
    _check_num_attention_heads(llm_dict["num_attention_heads"])
    raw_layer_types = _parse_raw_layer_types(llm_dict)
    layer_types = _parse_layer_types(llm_dict)
    attention_layer_types = _parse_attention_layer_types(
        llm_dict, llm_dict["num_hidden_layers"], model_type)
    if (not attention_layer_types and raw_layer_types
            and all(layer_type in _VALID_ATTENTION_LAYER_TYPES
                    for layer_type in raw_layer_types)):
        attention_layer_types = list(raw_layer_types)
    full_attention_only = (_is_gemma4_model_type(model_type)
                           and attention_layer_types
                           and all(layer_type == "full_attention"
                                   for layer_type in attention_layer_types))
    head_dim = llm_dict.get(
        "global_head_dim" if full_attention_only else "head_dim",
        llm_dict.get(
            "head_dim",
            llm_dict["hidden_size"] // llm_dict["num_attention_heads"]))
    global_head_dim = int(llm_dict.get("global_head_dim", 0) or 0)
    num_global_kv_heads = int(
        llm_dict.get("num_global_key_value_heads", 0) or 0)
    num_kv_heads = (
        num_global_kv_heads if full_attention_only and num_global_kv_heads else
        llm_dict.get("num_key_value_heads", llm_dict["num_attention_heads"]))
    dual_rope_configs = _get_dual_rope_configs(llm_dict)
    use_sw = llm_dict.get("use_sliding_window", False) or any(
        layer_type == "sliding_attention" for layer_type in raw_layer_types)
    sw_raw = llm_dict.get("sliding_window") if use_sw else None
    sliding_window_size = int(sw_raw) if sw_raw is not None else -1

    default_attention_scale_value = float(default_attention_scale(head_dim))
    default_mask_token_id = 4 if _is_gemma4_model_type(model_type) else 248070

    vocab_size = llm_dict.get("vocab_size", target_vocab_size)
    if vocab_size is None:
        raise ValueError("DFlash draft config must provide vocab_size")
    draft_vocab_size = llm_dict.get("draft_vocab_size")
    draft_vocab_size = int(vocab_size if draft_vocab_size is
                           None else draft_vocab_size)
    return ModelConfig(
        model_type=model_type,
        hidden_size=llm_dict["hidden_size"],
        num_hidden_layers=llm_dict["num_hidden_layers"],
        num_attention_heads=llm_dict["num_attention_heads"],
        num_key_value_heads=num_kv_heads,
        intermediate_size=llm_dict["intermediate_size"],
        head_dim=head_dim,
        global_head_dim=global_head_dim,
        num_global_key_value_heads=num_global_kv_heads,
        rms_norm_eps=llm_dict.get("rms_norm_eps", 1e-6),
        vocab_size=int(vocab_size),
        rope_theta=_get_rope_theta(llm_dict),
        max_position_embeddings=llm_dict.get("max_position_embeddings", 4096),
        default_attention_scale=default_attention_scale_value,
        rope_scaling=_select_rope_scaling(llm_dict),
        partial_rotary_factor=_get_partial_rotary_factor(llm_dict),
        hidden_activation=llm_dict.get("hidden_activation",
                                       llm_dict.get("hidden_act", "silu")),
        sliding_rope_config=dual_rope_configs.get("sliding_rope_config"),
        full_rope_config=dual_rope_configs.get("full_rope_config"),
        has_qk_norm=True,
        has_value_norm=_get_has_value_norm(llm_dict, model_type),
        attention_scaling=_get_attention_scaling(
            llm_dict, head_dim, default_attention_scale_value),
        attention_k_eq_v=bool(llm_dict.get("attention_k_eq_v", False)),
        final_logit_softcapping=llm_dict.get("final_logit_softcapping", None),
        torch_dtype=llm_dict.get("torch_dtype", "bfloat16"),
        tie_word_embeddings=False,
        sliding_window_size=sliding_window_size,
        layer_types=layer_types,
        attention_layer_types=attention_layer_types,
        raw_layer_types=raw_layer_types,
        rope_parameters=llm_dict.get("rope_parameters", None),
        draft_vocab_size=draft_vocab_size,
        reduced_vocab_size=(draft_vocab_size
                            if draft_vocab_size < int(vocab_size) else None),
        is_dflash_draft_flag=True,
        dflash_target_layer_ids=list(
            dflash_config.get(
                "target_layer_ids",
                llm_dict.get("target_layer_ids", [1, 8, 15, 22, 29]))),
        dflash_block_size=int(
            dflash_config.get("block_size", llm_dict.get("block_size", 16))),
        dflash_mask_token_id=int(
            dflash_config.get(
                "mask_token_id",
                llm_dict.get("mask_token_id", default_mask_token_id))),
        dflash_shift_label=shift_label,
        quant=quant,
    )


def make_dflash2_draft_config(
        draft_dir: str,
        default_attention_scale: Callable[[int], float]) -> ModelConfig:
    """Build and validate the production DFlash2 draft contract."""
    root_dict, llm_dict = load_checkpoint_config_dicts(draft_dir)
    resolved = resolve_dflash_contract(root_dict, llm_dict)
    if resolved.version != DFlashVersion.V2:
        raise ValueError(
            "DFlash2 draft checkpoint requires architecture DFlash2DraftModel")

    config = make_dflash_draft_config(draft_dir, default_attention_scale)
    if config.num_hidden_layers != 5:
        raise ValueError(
            "DFlash2 production checkpoint requires exactly five draft layers")

    return replace(
        config,
        dflash_version=resolved.version,
        dflash_target_layer_ids=list(resolved.target_layer_ids),
        dflash_block_size=resolved.block_size,
        dflash_mask_token_id=resolved.mask_token_id,
        is_dflash_draft_flag=True,
        dflash2_target_layer_ids=list(resolved.target_layer_ids),
        dflash2_block_size=resolved.block_size,
        dflash2_mask_token_id=resolved.mask_token_id,
        dflash2_is_causal=resolved.is_causal,
        dflash2_conv_kernel_size=resolved.conv_kernel_size,
        dflash2_conv_group_size=resolved.conv_group_size,
        dflash2_selector_rank=resolved.selector_rank,
        dflash2_selector_top_k=resolved.selector_top_k,
        dflash2_output_multiplier=resolved.output_multiplier,
        dflash2_final_logit_softcapping=resolved.final_logit_softcapping,
    )


def make_lilicorr_draft_config(
        draft_dir: str,
        default_attention_scale: Callable[[int], float]) -> ModelConfig:
    """Build and validate the LiLiCorr draft contract (DFlash2 + lattice head)."""
    root_dict, llm_dict = load_checkpoint_config_dicts(draft_dir)
    resolved = resolve_dflash_contract(root_dict, llm_dict)
    if not resolved.is_lilicorr:
        raise ValueError("LiLiCorr draft checkpoint requires "
                         "dflash_config.projector_type='lilicorr'")

    config = make_dflash2_draft_config(draft_dir, default_attention_scale)
    if config.draft_vocab_size != config.vocab_size:
        raise ValueError(
            "LiLiCorr candidates require the full target vocabulary: the draft "
            f"declares draft_vocab_size={config.draft_vocab_size}, "
            f"vocab_size={config.vocab_size}")
    head_hidden_size = resolved.lilicorr_hidden_size
    if head_hidden_size == config.hidden_size:
        raise ValueError(
            "LiLiCorr heads as wide as the draft hidden size "
            f"({config.hidden_size}) use an identity token_proj, which is not "
            "supported; set dflash_config.lilicorr_hidden_size to the head width"
        )
    quant = config.quant
    # LiLiCorr runs fc as the checkpoint's packed NVFP4 W4A16 GEMM. The draft
    # ships no lm_head: an `*lm_head` exclusion glob becomes the exact name, which
    # the base lm_head inheritance replaces when the base head is quantized.
    excluded = [
        name for name in quant.excluded
        if name != "fc" and not fnmatch.fnmatchcase("lm_head", name)
    ]
    if any(fnmatch.fnmatchcase("lm_head", name) for name in quant.excluded):
        excluded.append("lm_head")
    quant = replace(quant, excluded=excluded)

    dflash_config = llm_dict.get("dflash_config") or {}
    return replace(
        config,
        quant=quant,
        attention_sink_bias=bool(
            dflash_config.get("attention_sink_bias",
                              llm_dict.get("attention_sink_bias", False))),
        dflash_projector_type="lilicorr",
        lilicorr_candidate_topk=resolved.lilicorr_candidate_topk,
        lilicorr_hidden_size=head_hidden_size,
        lilicorr_num_layers=resolved.lilicorr_num_layers,
        lilicorr_num_heads=resolved.lilicorr_num_heads,
        lilicorr_mlp_ratio=resolved.lilicorr_mlp_ratio,
        lilicorr_factor_dim=resolved.lilicorr_factor_dim,
        lilicorr_vector_eps=resolved.lilicorr_vector_eps,
        lilicorr_logit_scale=resolved.lilicorr_logit_scale,
    )


def make_domino_draft_config(
        draft_dir: str,
        default_attention_scale: Callable[[int], float]) -> ModelConfig:
    """Build and validate the Domino draft contract (DFlash V1 + correction head)."""
    root_dict, llm_dict = load_checkpoint_config_dicts(draft_dir)
    resolved = resolve_dflash_contract(root_dict, llm_dict)
    if not resolved.is_domino:
        raise ValueError(
            "Domino draft checkpoint requires dflash_config.projector_type="
            "'domino'")

    config = make_dflash_draft_config(draft_dir, default_attention_scale)
    # The in-engine pool projection is a split of the checkpoint's FP16/BF16
    # embed_proj.0 and is filled by a load hook; keep it dense FP16.
    if "domino_pool.z_projection" not in config.quant.excluded:
        config.quant.excluded = list(
            config.quant.excluded) + ["domino_pool.z_projection"]

    return replace(
        config,
        dflash_version=resolved.version,
        dflash_target_layer_ids=list(resolved.target_layer_ids),
        dflash_block_size=resolved.block_size,
        dflash_mask_token_id=resolved.mask_token_id,
        is_dflash_draft_flag=True,
        dflash_projector_type="domino",
        domino_gru_hidden_dim=resolved.gru_hidden_dim,
        domino_proj_dim=resolved.proj_dim,
        domino_prefix_len=resolved.prefix_len,
        domino_pool_size=min(resolved.pool_size, config.vocab_size),
        domino_shift_label=resolved.shift_label,
    )


def make_jetspec_draft_config(
        draft_dir: str,
        default_attention_scale: Callable[[int], float]) -> ModelConfig:
    """Build a JetSpec draft config from an official JetSpec checkpoint.

    The public JetSpec Qwen3 checkpoint stores its metadata in ``dflash_config``
    because JetSpec reuses the DFlash draft-head implementation. Persist the
    exported runtime config under ``jetspec_config`` while also filling the
    DFlash fields consumed by :class:`DFlashDraftModel`.
    """
    _, llm_dict = load_checkpoint_config_dicts(draft_dir)
    base = make_dflash_draft_config(draft_dir, default_attention_scale)
    jetspec_config = (llm_dict.get("jetspec_config")
                      or llm_dict.get("dflash_config") or {})
    target_layer_ids = list(
        jetspec_config.get("target_layer_ids",
                           llm_dict.get("target_layer_ids", [])) or [])
    if not target_layer_ids:
        raise ValueError(
            "JetSpec draft config requires target_layer_ids in config.json.")

    block_size = int(
        jetspec_config.get("block_size",
                           llm_dict.get("block_size", base.dflash_block_size)))
    mask_token_id = int(
        jetspec_config.get("mask_token_id",
                           llm_dict.get("mask_token_id", 151669)))
    causal_head = bool(
        jetspec_config.get("causal_head", llm_dict.get("causal_head", True)))
    if not causal_head:
        raise ValueError(
            "JetSpec draft config requires causal_head=true; use DFlash for non-causal block drafts."
        )

    return replace(
        base,
        is_dflash_draft_flag=False,
        is_jetspec_draft_flag=True,
        jetspec_target_layer_ids=target_layer_ids,
        jetspec_block_size=block_size,
        jetspec_mask_token_id=mask_token_id,
        jetspec_causal_head=causal_head,
        dflash_target_layer_ids=target_layer_ids,
        dflash_block_size=block_size,
        dflash_mask_token_id=mask_token_id,
    )


def _parse_num_deepstack_features(
    config: dict,
    model_type: str,
    *,
    root_config: Optional[Dict[str, Any]] = None,
) -> int:
    """Return the number of deepstack visual features for ONNX / runtime.

    1. Explicit ``num_deepstack_features`` on the LLM (text) dict, then on
       the root dict.
    2. ``len(vision_config["deepstack_visual_indexes"])`` on the root config.
    3. Fallback: ``3`` for ``qwen3_vl`` / ``qwen3_omni`` when still unknown, else ``0``.
    """
    raw = config.get("num_deepstack_features")
    if raw is not None:
        return int(raw)

    if root_config is not None:
        raw_root = root_config.get("num_deepstack_features")
        if raw_root is not None:
            return int(raw_root)

        vision = root_config.get("vision_config")
        if isinstance(vision, dict):
            indexes = vision.get("deepstack_visual_indexes")
            if isinstance(indexes, (list, tuple)) and len(indexes) > 0:
                return len(indexes)

    root_mt = (root_config or {}).get("model_type") or ""
    # Strict equality, not substring — ``"qwen3_omni" in "qwen3_omni_next"``
    # is True and would otherwise report 3 deepstack features for Qwen3-Next Omni
    # (whose ``deepstack_visual_indexes`` is empty -> 0 features).
    if model_type in _DEEPSTACK_MODEL_TYPES or root_mt in _DEEPSTACK_MODEL_TYPES:
        return 3
    return 0


def _parse_accept_hidden_layer(
    config: dict,
    *,
    root_config: Optional[Dict[str, Any]] = None,
) -> int:
    """Return ``accept_hidden_layer`` (Thinker decoder layer count whose output
    is handed off to the Qwen3-Omni Talker), or -1 when not applicable.

    Lookup order:
      1. ``accept_hidden_layer`` at the LLM (text) dict top level.  This is
         the layout written by the standalone-Thinker quant export, which
         promotes the field out of ``talker_config`` into the Thinker root.
      2. ``root_config["talker_config"]["accept_hidden_layer"]`` for the
         full multimodal HF config.
      3. ``root_config["accept_hidden_layer"]`` as a defensive fallback for
         standalone Talker checkpoints where this field already lives at
         root.
    """
    raw = config.get("accept_hidden_layer")
    if raw is not None:
        return int(raw)
    if root_config is not None:
        talker_cfg = root_config.get("talker_config")
        if isinstance(talker_cfg, dict):
            raw = talker_cfg.get("accept_hidden_layer")
            if raw is not None:
                return int(raw)
        raw = root_config.get("accept_hidden_layer")
        if raw is not None:
            return int(raw)
    return -1


def _validate_mtp_constraints(
    *,
    model_type: str,
    mtp_num_hidden_layers: Optional[int],
    mtp_use_dedicated_embeddings: bool,
) -> None:
    """Validate the currently supported MTP config subset."""
    if mtp_num_hidden_layers is None and not mtp_use_dedicated_embeddings:
        return
    if (model_type or "").lower().startswith("nemotron_h"):
        if mtp_use_dedicated_embeddings:
            raise NotImplementedError(
                "Dedicated MTP embeddings are not supported for Nemotron-H MTP."
            )
        return
    if model_type not in _MTP_CONFIG_MODEL_TYPES:
        raise NotImplementedError(
            "MTP config parsing is only supported for Qwen3.5, dense Qwen3 "
            "and Nemotron-H checkpoints.")
    if mtp_num_hidden_layers != 1:
        raise NotImplementedError(
            "Only mtp_num_hidden_layers == 1 is supported for MTP.")
    if mtp_use_dedicated_embeddings:
        raise NotImplementedError(
            "Dedicated MTP embeddings are not supported for MTP.")


# Nested ``eagle_config`` aux IDs name decoder-layer outputs; layer j's output
# is HF hidden-state index j + 1.
_EAGLE_CONFIG_LAYER_ID_OFFSET = 1
_EAGLE_DRAFT_FEATURE_KEYS = ("fc_norm", "norm_output",
                             "efficient_lm_head_config")
# Names used in errors where the checkpoint key is not self-explanatory.
_EAGLE_DRAFT_FEATURE_LABELS = {
    "efficient_lm_head_config": "a low-rank candidate head"
}
# Checkpoint prefixes of the low-rank candidate head factors and the
# Eagle4DraftModel modules they load into.
_EAGLE_CANDIDATE_HEAD_KEY_PREFIXES = (
    ("efficient_lm_head_down_proj.", "candidate_head.down_proj."),
    ("efficient_lm_head.", "candidate_head.up_proj."),
)
# EAGLE drafts read the base embedding.
_EAGLE_SKIPPED_CHECKPOINT_KEYS = frozenset({"embed_tokens.weight"})
# Shapes supported by LowRankCandidateHeadPlugin.
_MAX_DRAFT_CANDIDATE_COUNT = 256
_MAX_DRAFT_CANDIDATE_RANK = 256
_DRAFT_CANDIDATE_RANK_MULTIPLE = 16


@dataclass(frozen=True)
class _EagleDraftFeatures:
    fc_norm: bool = False
    norm_output: bool = False
    candidate_rank: int = 0
    candidate_count: int = 0


def _eagle_draft_feature_keys(llm_dict: Dict[str, Any]) -> List[str]:
    """Return the EAGLE draft capability keys ``llm_dict`` turns on.

    A key that is false, null or empty does not count.
    """
    return [key for key in _EAGLE_DRAFT_FEATURE_KEYS if llm_dict.get(key)]


def _describe_eagle_draft_features(llm_dict: Dict[str, Any]) -> str:
    """Name the EAGLE draft capabilities ``llm_dict`` turns on ("" for none)."""
    features = [
        _EAGLE_DRAFT_FEATURE_LABELS.get(key, key)
        for key in _eagle_draft_feature_keys(llm_dict)
    ]
    if len(features) > 1:
        return f"{', '.join(features[:-1])} and {features[-1]}"
    return features[0] if features else ""


def _parse_eagle_target_layer_ids(
        llm_dict: Dict[str, Any],
        num_layers: Optional[int] = None) -> List[int]:
    """Resolve an EAGLE draft's aux hidden-state layers as HF indices.

    Precedence: top-level ``eagle_aux_hidden_state_layer_ids`` (HF index,
    as-is), then nested ``eagle_config.eagle_aux_hidden_state_layer_ids``
    (layer outputs), then flat ``target_layer_ids`` (unchanged), else ``[]`` so
    the base applies its default. ``num_layers`` is the base's decoder-layer
    count, when known.
    """
    top_level = llm_dict.get("eagle_aux_hidden_state_layer_ids")
    if top_level:
        return list(top_level)
    eagle_config = llm_dict.get("eagle_config") or {}
    if eagle_config.get("use_aux_hidden_state") is False:
        raise ValueError(
            "EAGLE draft sets eagle_config.use_aux_hidden_state=false; only "
            "drafts that fuse base aux hidden states are supported.")
    nested = eagle_config.get("eagle_aux_hidden_state_layer_ids")
    if nested:
        layer_outputs = [int(layer) for layer in nested]
        if any(layer < 0 for layer in layer_outputs):
            raise ValueError(
                "eagle_config.eagle_aux_hidden_state_layer_ids must be "
                f"non-negative, got {layer_outputs}.")
        # TorchSpec concatenates aux states in list order, so sorting would
        # permute the fc input columns.
        if ("_torchspec_version" in llm_dict
                and layer_outputs != sorted(layer_outputs)):
            raise ValueError(
                "TorchSpec eagle_config.eagle_aux_hidden_state_layer_ids "
                f"must be ascending, got {layer_outputs}.")
        layer_ids = sorted(layer + _EAGLE_CONFIG_LAYER_ID_OFFSET
                           for layer in layer_outputs)
        # HF index num_layers is the final-norm output, not a layer output.
        if num_layers is not None and any(layer >= num_layers
                                          for layer in layer_ids):
            raise ValueError(
                f"EAGLE aux layer outputs {layer_outputs} need HF indices "
                f"{layer_ids} below the base num_hidden_layers={num_layers}.")
        return layer_ids
    return list(llm_dict.get("target_layer_ids") or [])


def _parse_eagle_draft_features(
        llm_dict: Dict[str, Any],
        target_layer_ids: List[int]) -> _EagleDraftFeatures:
    """Parse and validate the optional EAGLE draft norms and candidate head."""
    fc_norm = bool(llm_dict.get("fc_norm", False))
    norm_output = bool(llm_dict.get("norm_output", False))
    for key in ("norm_before_fc", "norm_before_residual"):
        if llm_dict.get(key):
            raise ValueError(
                "EAGLE drafts with fc_norm, norm_output or a candidate head do "
                f"not support {key}.")
    if _is_gemma4_model_type(llm_dict.get("model_type", "")):
        raise ValueError(
            "EAGLE drafts with fc_norm, norm_output or a candidate head do not "
            "support a Gemma4 draft layer.")
    num_aux_hidden_states = llm_dict.get("num_aux_hidden_states")
    if (num_aux_hidden_states is not None and target_layer_ids
            and int(num_aux_hidden_states) != len(target_layer_ids)):
        raise ValueError(
            f"EAGLE draft num_aux_hidden_states={num_aux_hidden_states} "
            f"does not match its {len(target_layer_ids)} aux layer IDs "
            f"{target_layer_ids}.")
    head = llm_dict.get("efficient_lm_head_config")
    if not head:
        return _EagleDraftFeatures(fc_norm=fc_norm, norm_output=norm_output)

    head_type = head.get("type")
    if head_type != "low_rank":
        raise ValueError(
            f"EAGLE candidate head type {head_type!r} is not supported; "
            "expected 'low_rank'.")
    vocab_size = int(llm_dict["vocab_size"])
    count = int(head.get("num_candidates", 0) or 0)
    max_count = min(vocab_size, _MAX_DRAFT_CANDIDATE_COUNT)
    if not 0 < count <= max_count:
        raise ValueError(
            f"EAGLE candidate head num_candidates={count} must be in "
            f"[1, {max_count}] (vocab_size={vocab_size}).")
    rank = int(head.get("rank", 0) or 0)
    if not 0 < rank <= _MAX_DRAFT_CANDIDATE_RANK:
        raise ValueError(f"EAGLE candidate head rank={rank} must be in "
                         f"[1, {_MAX_DRAFT_CANDIDATE_RANK}].")
    if rank % _DRAFT_CANDIDATE_RANK_MULTIPLE:
        raise ValueError(
            f"EAGLE candidate head rank={rank} must be a multiple of "
            f"{_DRAFT_CANDIDATE_RANK_MULTIPLE}.")
    # The engine's only hidden output is the final-norm output, which both the
    # head and the next draft step read.
    if not norm_output:
        raise ValueError("An EAGLE candidate head requires norm_output=true.")
    if llm_dict.get("final_logit_softcapping") is not None:
        raise ValueError(
            "An EAGLE candidate head does not support final_logit_softcapping."
        )
    draft_vocab_size = llm_dict.get("draft_vocab_size")
    if draft_vocab_size is not None and int(draft_vocab_size) != vocab_size:
        raise ValueError("An EAGLE candidate head scores the full vocabulary; "
                         f"draft_vocab_size={draft_vocab_size} must equal "
                         f"vocab_size={vocab_size}.")
    return _EagleDraftFeatures(fc_norm=fc_norm,
                               norm_output=norm_output,
                               candidate_rank=rank,
                               candidate_count=count)


def require_dense_eagle_draft(llm_dict: Dict[str, Any], consumer: str) -> None:
    """Raise unless an EAGLE draft config describes the dense EAGLE-3 graph.

    ``fc_norm``, ``norm_output`` and low-rank candidate heads need an EAGLE-4
    draft graph. ``consumer`` implements the dense graph alone and must not
    silently treat such a draft as dense. A key that is false, null or empty
    does not count.
    """
    listed = _describe_eagle_draft_features(llm_dict)
    if listed:
        raise NotImplementedError(
            f"{consumer} implements only the dense EAGLE3 draft and does not "
            f"support {listed}. Export this draft with "
            "tensorrt-edgellm-export instead.")


def _parse_raw_layer_types(config: dict) -> List[str]:
    """Return checkpoint layer type strings without canonicalization."""
    raw = config.get("layers_block_type") or config.get("layer_types")
    if raw is None:
        return []
    return [str(layer_type) for layer_type in raw]


def _is_nemotron_h_config(config: dict) -> bool:
    return (config.get("model_type") or "").lower().startswith("nemotron_h")


def _canonical_layer_type(block_type: str, is_nemotron_h: bool) -> str:
    """Map one checkpoint block-type name onto a canonical layer label.

    ``"linear_attention"`` is ambiguous: it covers any sub-quadratic mixer, so
    it denotes Mamba2 for NemotronH and GatedDeltaNet for Qwen3.5. The model
    family therefore resolves it, along with ``"full_attention"``, which Gemma4
    keeps verbatim for per-layer head_dim dispatch.
    """
    bt = str(block_type).lower()
    if bt == "linear_attention":
        return LAYER_MAMBA if is_nemotron_h else LAYER_GDN
    if "mamba" in bt:
        return LAYER_MAMBA
    if bt == "moe":
        return LAYER_MOE
    if "mlp" in bt:
        return LAYER_MLP
    if bt in _VALID_ATTENTION_LAYER_TYPES:
        return LAYER_ATTN if is_nemotron_h else bt
    return LAYER_ATTN


def _parse_mtp_layer_types(config: dict) -> List[str]:
    """Return the MTP draft stack's per-layer block types.

    NemotronH declares the stack either as ``mtp_layers_block_type`` (a list,
    which transformers >= 5.14 rewrites into the ``linear_attention`` /
    ``full_attention`` spelling) or as the legacy ``mtp_hybrid_override_pattern``
    string (e.g. ``"*E"``).
    """
    raw = config.get("mtp_layers_block_type")
    if raw:
        is_nemotron_h = _is_nemotron_h_config(config)
        return [_canonical_layer_type(bt, is_nemotron_h) for bt in raw]
    pattern = config.get("mtp_hybrid_override_pattern") or ""
    return [
        _HYBRID_PATTERN_MAP[ch] for ch in pattern if ch in _HYBRID_PATTERN_MAP
    ]


def _parse_layer_types(config: dict) -> List[str]:
    """Return per-layer block type list from config.

    Reads ``layers_block_type`` or ``layer_types`` directly if present, each
    entry resolved by :func:`_canonical_layer_type`. For models using
    ``hybrid_override_pattern`` (e.g. NemotronH), parses the pattern string
    where ``M`` = mamba, ``-`` = mlp, ``*`` = attention. Falls back to all
    attention layers.
    """
    raw = config.get("layers_block_type") or config.get("layer_types")
    if raw is not None:
        is_nemotron_h = _is_nemotron_h_config(config)
        return [_canonical_layer_type(bt, is_nemotron_h) for bt in raw]
    pattern = config.get("hybrid_override_pattern")
    if pattern is not None:
        return [
            _HYBRID_PATTERN_MAP[ch] for ch in pattern
            if ch in _HYBRID_PATTERN_MAP
        ]
    n = config["num_hidden_layers"]
    return [LAYER_ATTN] * n


def _parse_puzzle_block_configs(config: dict,
                                layer_types: List[str]) -> List[dict]:
    """Keep the heterogeneous MoE geometry from Nemotron-H Puzzle."""
    if config.get("model_type") != "nemotron_h_puzzle":
        return []
    blocks = config.get("block_configs")
    if not isinstance(blocks, list) or len(blocks) != len(layer_types):
        raise ValueError(
            "Nemotron-H Puzzle requires one block_config per layer")
    result = []
    for index, (block, layer_type) in enumerate(zip(blocks, layer_types)):
        if not isinstance(block, dict) or _canonical_layer_type(
                block.get("block_type", ""), True) != layer_type:
            raise ValueError(
                f"Invalid Nemotron-H Puzzle block_config at layer {index}")
        if layer_type == LAYER_MOE:
            geometry = {
                name: int(block.get(name, 0) or 0)
                for name in ("n_routed_experts", "moe_intermediate_size",
                             "num_experts_per_tok")
            }
            if any(value <= 0 for value in geometry.values()):
                raise ValueError(
                    f"Incomplete Nemotron-H Puzzle MoE geometry at layer {index}"
                )
            result.append(geometry)
        else:
            result.append({})
    return result


def _parse_mamba_cfg(config: dict,
                     layer_types: List[str],
                     model_dir: str = "") -> Optional[MambaConfig]:
    """Return a MambaConfig if any layer is a mamba layer, else None.

    ``model_dir`` is used to detect ``conv_dim`` from the actual checkpoint
    weight shape when the config does not declare it explicitly.  This resolves
    a circular dependency: ``conv_dim`` depends on ``n_groups`` and vice-versa.
    """
    if LAYER_MAMBA not in layer_types:
        return None

    num_heads = config.get("mamba_num_heads", 0)
    head_dim = config.get("mamba_head_dim", 0)
    ssm_state_size = config.get("ssm_state_size", 0)
    conv_kernel = config.get("conv_kernel", config.get("mamba_d_conv", 4))

    d_inner = num_heads * head_dim
    n_groups = config.get(
        "n_groups",
        config.get("mamba_n_groups", config.get("mamba_num_groups", 1)))

    if "conv_dim" in config:
        conv_dim = config["conv_dim"]
    else:
        # Try to read conv_dim from the actual checkpoint shape to avoid
        # circular dependency with n_groups when neither key is in config.
        detected = _detect_mamba_conv_dim(model_dir) if model_dir else 0
        if detected > 0:
            conv_dim = detected
        else:
            conv_dim = d_inner + 2 * n_groups * ssm_state_size

    # Back-derive n_groups from conv_dim if not explicit in config
    if n_groups == 1 and conv_dim > d_inner:
        derived = (conv_dim - d_inner) // (2 * ssm_state_size)
        n_groups = derived if derived > 0 else 1

    return MambaConfig(
        num_heads=num_heads,
        head_dim=head_dim,
        ssm_state_size=ssm_state_size,
        conv_dim=conv_dim,
        conv_kernel=conv_kernel,
        n_groups=n_groups,
    )


def _detect_mamba_conv_dim(model_dir: str) -> int:
    """Return conv_dim by reading the first Mamba conv1d.weight shape from the checkpoint.

    The conv1d weight has shape ``[conv_dim, 1, conv_kernel]`` so ``shape[0]``
    gives ``conv_dim`` directly — no tensor data is loaded.  Returns 0 if
    not found (caller falls back to formula-based derivation).
    """
    try:
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            weight_map: dict = index.get("weight_map", {})
            shard_for_key: Optional[str] = None
            target_key: Optional[str] = None
            for k, shard in weight_map.items():
                if k.endswith(".mixer.conv1d.weight"):
                    shard_for_key = shard
                    target_key = k
                    break
            if shard_for_key and target_key:
                shard_path = os.path.join(model_dir, shard_for_key)
                from safetensors import safe_open
                with safe_open(shard_path, framework="pt") as f:
                    return f.get_slice(target_key).get_shape()[0]
        else:
            single_path = os.path.join(model_dir, "model.safetensors")
            if os.path.exists(single_path):
                from safetensors import safe_open
                with safe_open(single_path, framework="pt") as f:
                    for k in f.keys():
                        if k.endswith(".mixer.conv1d.weight"):
                            return f.get_slice(k).get_shape()[0]
    except (OSError, KeyError, ImportError):
        pass
    return 0


def _parse_gdn_cfg(config: dict,
                   layer_types: List[str]) -> Optional[GdnConfig]:
    """Return a GdnConfig if any layer is a GDN (linear_attention) layer, else None."""
    if LAYER_GDN not in layer_types:
        return None
    return GdnConfig(
        num_key_heads=config.get("linear_num_key_heads", 0),
        num_value_heads=config.get("linear_num_value_heads", 0),
        key_head_dim=config.get("linear_key_head_dim", 0),
        value_head_dim=config.get("linear_value_head_dim", 0),
        conv_kernel=config.get("linear_conv_kernel_dim", 4),
        output_gate_type=(config.get("output_gate_type")
                          or config.get("hidden_act", "silu")),
    )


def _get_partial_rotary_factor(llm_dict: Dict[str, Any]) -> float:
    """Extract partial_rotary_factor from config dict.

    Qwen3.5 stores this inside ``rope_parameters`` rather than at top level.
    """
    prf = llm_dict.get("partial_rotary_factor")
    if prf is not None:
        return float(prf)
    for key in ("rope_parameters", "rope_scaling"):
        nested = llm_dict.get(key)
        if not isinstance(nested, dict):
            continue
        if nested.get("partial_rotary_factor") is not None:
            return float(nested["partial_rotary_factor"])
        for attention_type in ("full_attention", "sliding_attention"):
            attention_params = nested.get(attention_type)
            if (isinstance(attention_params, dict) and
                    attention_params.get("partial_rotary_factor") is not None):
                return float(attention_params["partial_rotary_factor"])
    return 1.0


def _detect_has_qk_norm(model_dir: str, model_type: str = "") -> bool:
    """Detect QK-norm by scanning checkpoint key names.

    ``*.q_norm.weight`` (Qwen3 convention) is model-agnostic. The HunYuan V1
    ``*.query_layernorm.weight`` spelling is only honored for model families
    whose loader remaps it onto ``q_norm`` (see ``_hunyuan_key_remap``);
    other families using that key name for unrelated norms must not trip
    the fused QK-norm path.
    """
    keys = _checkpoint_weight_keys(model_dir)
    if any(".q_norm.weight" in k for k in keys):
        return True
    return (model_type in _QUERY_LAYERNORM_KEY_MODEL_TYPES
            and any(".query_layernorm.weight" in k for k in keys))


def _checkpoint_weight_keys(model_dir: str) -> List[str]:
    """Return checkpoint tensor keys, ignoring stale shard indexes when needed."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(index_path):
        try:
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            missing_shards = {
                shard
                for shard in set(weight_map.values())
                if not os.path.exists(os.path.join(model_dir, shard))
            }
            if not missing_shards or not os.path.exists(single_path):
                return list(weight_map.keys())
        except (OSError, json.JSONDecodeError):
            pass

    if os.path.exists(single_path):
        try:
            from safetensors import safe_open
            with safe_open(single_path, framework="pt") as f:
                return list(f.keys())
        except (OSError, ImportError):
            pass
    return []


_VL_LLM_PREFIXES = ("language_model.", "text_model.", "llm.", "thinker.")


def _strip_vl_prefix(name: str) -> str:
    """Strip a known VL wrapper prefix (e.g. ``language_model.``) if present."""
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model."):]
    if name.startswith("model.visual."):
        return "visual." + name[len("model.visual."):]
    for prefix in _VL_LLM_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _normalize_module_name(name: str) -> str:
    """Normalise a checkpoint / hf_quant_config module name to the short
    namespace that ``make_linear`` uses (``layers.N...``, ``lm_head``, etc.).

    LLM ``make_linear`` callers pass names without any ``model.`` prefix
    (see ``modeling_default.py``: ``module_name=f"layers.{i}.mlp.gate_proj"``).
    For multimodal checkpoints whose keys are
    ``model.language_model.layers.N...`` the entire compound prefix must be
    stripped so the resulting short name matches what ``module_quant_type``
    looks up.  Single VL prefixes and bare ``model.`` follow the same rule.

    Used by ``_effective_excluded_modules``, ``_detect_modelopt_unquantized_linears``,
    and ``_parse_mixed_precision`` so that ``excluded``, ``layer_overrides``,
    and ``module_name`` all share the same name space.
    """
    if name.startswith("model.language_model."):
        return name[len("model.language_model."):]
    if name.startswith("thinker.model."):
        return name[len("thinker.model."):]
    # Talker sub-LLM keys share the thinker short-name space after
    # ``_make_sub_model_dir(key_prefix='talker.')`` staging.
    if name.startswith("talker.model."):
        return name[len("talker.model."):]
    # Bare submodel prefixes (consolidated Qwen3-Omni roots): the visual /
    # audio / code_predictor builders use short names without them.
    if name.startswith("thinker."):
        return name[len("thinker."):]
    if name.startswith("talker."):
        return name[len("talker."):]
    if name.startswith("model.decoder."):
        return name[len("model.decoder."):]
    for prefix in _VL_LLM_PREFIXES + ("model.", ):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _detect_unquantized_modules(model_dir: str) -> List[str]:
    """Return module names whose weights are plain float (not int4 quantized).

    Some layers (often ``lm_head``) use ``*.weight`` instead of ``*.qweight``.
    Checkpoint wrapper prefixes (``model.``, ``language_model.``, etc.) are
    stripped so that the returned names match the short names used by
    ``make_linear()``.
    """
    all_keys = _checkpoint_weight_keys(model_dir)

    # Top-level module prefix = everything before the last dot segment
    qweight_modules = {
        k.rsplit(".", 1)[0]
        for k in all_keys if k.endswith(".qweight")
    }
    weight_modules = {
        k.rsplit(".", 1)[0]
        for k in all_keys if k.endswith(".weight")
    }
    excluded: List[str] = [
        _normalize_module_name(m) for m in (weight_modules - qweight_modules)
    ]
    # lm_head may have neither .weight nor .qweight when tie_word_embeddings=True
    # (the checkpoint omits lm_head.weight entirely).  Treat it as FP16 so that
    # tie_weights() can clone embed_tokens.weight into it after loading.
    all_linear_stripped = {
        _normalize_module_name(m)
        for m in qweight_modules | weight_modules
    }
    if "lm_head" not in all_linear_stripped:
        excluded.append("lm_head")
    return _with_gdn_fused_exclusions(excluded)


_GDN_INPUT_PROJ_MODULES = ("in_proj_qkv", "in_proj_z", "in_proj_b",
                           "in_proj_a")

# Fused HF GDN projection names (as they appear in a ModelOpt ``ignore`` /
# ``modules_to_not_convert`` list) mapped to the split projections
# trt-edge-llm actually builds. The checkpoint loader splits
# ``in_proj_qkvz`` -> ``in_proj_qkv`` / ``in_proj_z`` and ``in_proj_ba`` ->
# ``in_proj_b`` / ``in_proj_a`` (Qwen3-Next family). Excluding only the fused
# name would otherwise leave the split Linears at the dominant quant type
# (e.g. NVFP4) even though their weights are plain FP16, so ``make_linear``
# would build an NVFP4Linear against an unquantized weight.
_GDN_FUSED_PROJ_SPLITS: Dict[str, Tuple[str, ...]] = {
    "in_proj_qkvz": ("in_proj_qkv", "in_proj_z"),
    "in_proj_ba": ("in_proj_b", "in_proj_a"),
}


def _with_gdn_fused_exclusions(modules: List[str]) -> List[str]:
    """Reconcile GDN input-projection exclusions with the split/fused forms.

    Expands fused HF names (``in_proj_qkvz`` / ``in_proj_ba``) present in the
    ignore list into the split projections the model builds, and adds the
    synthetic ``in_proj_fused`` when all four split projections are FP16.
    """
    result = set(modules)
    # Fused HF name -> split projections (so an excluded ``in_proj_qkvz``
    # also excludes the ``in_proj_qkv`` / ``in_proj_z`` the model builds).
    for module in list(result):
        for fused, splits in _GDN_FUSED_PROJ_SPLITS.items():
            suffix = f".{fused}"
            if module.endswith(suffix):
                prefix = module[:-len(suffix)]
                for split in splits:
                    result.add(f"{prefix}.{split}")
                break
    by_prefix: Dict[str, set] = {}
    for module in result:
        for proj in _GDN_INPUT_PROJ_MODULES:
            suffix = f".{proj}"
            if module.endswith(suffix):
                by_prefix.setdefault(module[:-len(suffix)], set()).add(proj)
                break
    for prefix, projs in by_prefix.items():
        if all(proj in projs for proj in _GDN_INPUT_PROJ_MODULES):
            result.add(f"{prefix}.in_proj_fused")
    return sorted(result)


def _detect_quantized_modules(model_dir: str) -> List[str]:
    """Return modules that have checkpoint quantization sidecars."""
    all_keys = _checkpoint_weight_keys(model_dir)

    suffixes = (".qweight", ".weight_scale", ".weight_scale_2", ".input_scale",
                ".scales")
    modules = {
        _strip_vl_prefix(k.rsplit(".", 1)[0])
        for k in all_keys if k.endswith(suffixes)
    }
    return sorted(modules)


def _effective_excluded_modules(model_dir: str,
                                excluded: List[str]) -> List[str]:
    """Drop exclusions contradicted by quantized tensors in the checkpoint,
    and normalize remaining names so they match what ``make_linear`` looks up.

    Normalisation mirrors ``_parse_mixed_precision``'s strip on
    ``layer_overrides`` keys (``language_model.`` / ``text_model.`` / ``llm.``
    / ``model.``) so a checkpoint that writes ``model.visual.blocks.X.Y`` to
    ``exclude_modules`` matches the ``visual.blocks.X.Y`` ``module_name`` the
    modeling code passes.
    """
    quantized_modules = set(_detect_quantized_modules(model_dir))
    normalized = [
        _normalize_module_name(module) for module in excluded
        if _strip_vl_prefix(module) not in quantized_modules
    ]
    return _with_gdn_fused_exclusions(normalized)


def _detect_modelopt_unquantized_linears(model_dir: str,
                                         submodel_prefix: str = ""
                                         ) -> List[str]:
    """Return module_name strings of Linears the ModelOpt checkpoint left unquantized.

    A ModelOpt-quantized Linear stores both ``<name>.weight`` (packed) and
    ``<name>.weight_scale`` (scale tensor; FP8 / NVFP4 / MXFP8 / AWQ / INT8-SQ
    all emit this).  Linears that ModelOpt skipped — typically because their
    wildcard had ``enable: False`` (visual / audio / lm_head) — only have
    ``<name>.weight``.

    We compute the set of "has .weight without .weight_scale" modules from the
    checkpoint index, then return them in the short form ``make_linear`` uses
    (leading ``model.`` and VL wrapper prefixes stripped).  Norm and embedding
    names also fall into this set but are harmless: ``make_linear`` only
    consults ``excluded`` for paths that actually go through it (i.e. real
    Linears), so extra entries are inert.

    Used by ``_parse_quant`` to plug a long-standing gap: for dominant-quant
    checkpoints (``quant_algo: FP8 / NVFP4 / W4A16_AWQ / ...``) ModelOpt does
    NOT populate ``exclude_modules`` even when whole submodules (visual tower,
    audio encoder) were skipped during PTQ.  Without this augmentation,
    ``make_linear``'s dominant fallback would build NVFP4Linear / FP8Linear
    against FP16 weights → shape mismatch on export.

    Complements ``_effective_excluded_modules`` (which drops false-positive
    excludes when the checkpoint contradicts them).  This helper supplies the
    opposite direction: false-negative excludes when ``exclude_modules`` is
    silent on a submodule that was actually skipped during PTQ.
    """
    all_keys = _checkpoint_weight_keys(model_dir)

    if not all_keys:
        return []

    # Omni roots hold every sub-model's tensors. Restrict the scan, or another
    # sub-model's unquantized Linears would be normalised onto this one's
    # module names and silently unquantize them.
    if submodel_prefix:
        all_keys = [k for k in all_keys if k.startswith(submodel_prefix)]

    weight_modules = {
        k.rsplit(".", 1)[0]
        for k in all_keys if k.endswith(".weight")
    }
    scale_modules = {
        k.rsplit(".", 1)[0]
        for k in all_keys if k.endswith(".weight_scale")
    }
    unquantized = weight_modules - scale_modules

    excluded = set()
    for name in unquantized:
        normalized = _normalize_module_name(name)
        if normalized.endswith(".self_attn.qkv_proj"):
            prefix = normalized[:-len("qkv_proj")]
            excluded.update(f"{prefix}{proj}"
                            for proj in ("q_proj", "k_proj", "v_proj"))
        elif normalized.endswith(".mlp.gate_up_proj"):
            prefix = normalized[:-len("gate_up_proj")]
            excluded.update(f"{prefix}{proj}"
                            for proj in ("gate_proj", "up_proj"))
        else:
            excluded.add(normalized)

    # Normalise to the same name space ``layer_overrides`` and ``excluded``
    # use, so ``make_linear`` finds entries via its ``module_name`` lookup.
    return _with_gdn_fused_exclusions(sorted(excluded))


def _scope_exclusions(patterns: List[str], submodel_prefix: str) -> List[str]:
    """Keep only exclusions that apply to *submodel_prefix*, prefix stripped.

    Omni roots share one ignore list across thinker/talker, and short-name
    normalisation later drops both prefixes. A talker-scoped glob such as
    ``talker.model.layers.0.mlp.shared_expert*`` would therefore also unquantize
    the thinker's own shared_expert. A leading ``*`` marks a submodel-agnostic
    glob and is kept as-is.
    """
    if not submodel_prefix:
        return patterns
    scoped = []
    for pat in patterns:
        if pat.startswith(submodel_prefix):
            scoped.append(pat[len(submodel_prefix):])
        elif pat.startswith("*") or pat == "":
            scoped.append(pat)
    return scoped


def _grouped_quant_type(group: dict, root_format: str,
                        quant_method: str) -> Optional[str]:
    group_format = str(group.get("format", root_format)).lower()
    weights = group.get("weights") or {}
    inputs = group.get("input_activations") or {}
    # NVFP4 weights without quantized activations execute as W4A16.
    if "nvfp4" in group_format:
        return QUANT_NVFP4 if inputs else QUANT_NVFP4_A16
    if (str(weights.get("type", "")).lower() == "float"
            and int(weights.get("num_bits", 0) or 0) == 4 and not inputs):
        return QUANT_NVFP4_A16
    # ModelOpt writes weight-only int4 groups only for W4A16_AWQ (uint8
    # packed); compressed-tensors int4 uses another packing.
    if (quant_method == "modelopt" and not inputs
            and str(weights.get("type", "")).lower() == "int"
            and int(weights.get("num_bits", 0) or 0) == 4):
        return QUANT_INT4_AWQ_MODELOPT
    if (str(weights.get("type", "")).lower() == "float"
            and str(inputs.get("type", "")).lower() == "float"):
        weight_bits = int(weights.get("num_bits", 0) or 0)
        input_bits = int(inputs.get("num_bits", 0) or 0)
        if weight_bits == 4 and input_bits == 4:
            return QUANT_NVFP4
        if weight_bits == 8 and input_bits == 8:
            return QUANT_FP8
    return None


def _parse_grouped_nvfp4(model_dir: str,
                         config: dict,
                         submodel_prefix: str = "") -> Optional[QuantConfig]:
    qc = config.get("quantization_config")
    if not isinstance(qc, dict):
        return None
    quant_method = str(qc.get("quant_method", "")).lower()
    if quant_method not in ("compressed-tensors", "modelopt"):
        return None

    config_groups = qc.get("config_groups") or {}
    root_format = str(qc.get("format", ""))
    groups = [(group, _grouped_quant_type(group, root_format, quant_method))
              for group in config_groups.values()]
    if not any(quant_type in (QUANT_NVFP4, QUANT_NVFP4_A16)
               for _, quant_type in groups):
        return None
    # Groups naming another algorithm (e.g. Q8_0, IQ2_XXS) are described by
    # the per-layer metadata, which the full parsers read.
    if any(quant_type is None and group.get("quant_algo")
           for group, quant_type in groups):
        return None

    quantized_layers = {}
    global_types = []
    for group, quant_type in groups:
        if quant_type is None:
            continue
        group_size = int((group.get("weights") or {}).get(
            "group_size",
            16 if quant_type in (QUANT_NVFP4, QUANT_NVFP4_A16) else 1))
        # _parse_mixed_precision re-reads quant_algo as a ModelOpt algo name.
        # A grouped FP8 weight is block-scaled.
        algo = quant_type
        if quant_type == QUANT_FP8 and group_size > 1:
            algo = "FP8_BLOCK_SCALES"
        elif quant_type == QUANT_INT4_AWQ_MODELOPT:
            algo = "W4A16_AWQ"
        for target in group.get("targets") or ():
            if target == "Linear":
                global_types.append((quant_type, group_size))
            else:
                quantized_layers[target] = {
                    "quant_algo": algo,
                    "group_size": group_size,
                }

    excluded = _effective_excluded_modules(
        model_dir,
        _scope_exclusions(list(qc.get("ignore", [])), submodel_prefix))
    if quantized_layers:
        num_experts, top_k = _moe_routing(config)
        dominant, group_size, overrides, layer_group_sizes = (
            _parse_mixed_precision(quantized_layers,
                                   num_experts=num_experts,
                                   top_k=top_k))
        # Named groups can retain targets left unquantized in the checkpoint.
        # Reconcile only those overrides with their serialized weight payloads.
        excluded.extend(name for name in _detect_modelopt_unquantized_linears(
            model_dir, submodel_prefix)
                        if name in overrides and name not in excluded)
        return QuantConfig(
            quant_type=dominant,
            group_size=group_size,
            kv_cache_quant=_compressed_kv_cache_quant(qc),
            excluded=excluded,
            layer_overrides=overrides,
            layer_group_sizes=layer_group_sizes,
            is_mixed_precision=True,
        )

    quant_type, group_size = (global_types[0] if global_types else
                              (QUANT_NVFP4, 16))
    quant_type = _select_nvfp4_quant_type(quant_type)
    excluded.extend(module for module in _detect_modelopt_unquantized_linears(
        model_dir, submodel_prefix) if module not in excluded)
    return QuantConfig(
        quant_type=quant_type,
        group_size=group_size,
        kv_cache_quant=_compressed_kv_cache_quant(qc),
        excluded=excluded,
    )


def _parse_compressed_integer(
        model_dir: str,
        config: dict,
        submodel_prefix: str = "") -> Optional[QuantConfig]:
    """Lower uniform compressed-tensors integer weights to A16 execution."""
    qc = config.get("quantization_config")
    if not isinstance(qc, dict) or str(qc.get(
            "quant_method",
            "")).lower() not in ("compressed-tensors", "compressed_tensors"):
        return None

    contracts = []
    for group in (qc.get("config_groups") or {}).values():
        weights = group.get("weights") or {}
        if str(weights.get("type", "")).lower() != "int":
            return None
        # Activation observers do not change the serialized integer weight
        # layout. Edge-LLM consumes the same weights through W4A16/W8A16 when
        # the checkpoint's W4A4/W8A8 execution contract is unavailable.
        bits = int(weights.get("num_bits") or 0)
        if bits not in (4, 8):
            raise ValueError(
                f"unsupported compressed-tensors integer width {bits}; "
                "expected 4 or 8")
        quant_type = QUANT_INT4_GPTQ if bits == 4 else QUANT_INT8_GPTQ
        group_size = int(
            weights.get("group_size") or (128 if bits == 4 else 0))
        contracts.append((quant_type, group_size))

    if not contracts:
        raise ValueError(
            "unsupported compressed-tensors checkpoint: config_groups is empty"
        )
    if len(set(contracts)) != 1:
        raise ValueError(
            "mixed compressed-tensors integer layouts are not supported")

    quant_type, group_size = contracts[0]
    excluded = _effective_excluded_modules(
        model_dir,
        _scope_exclusions(list(qc.get("ignore", [])), submodel_prefix))
    excluded.extend(module for module in _detect_modelopt_unquantized_linears(
        model_dir, submodel_prefix) if module not in excluded)
    return QuantConfig(
        quant_type=quant_type,
        group_size=group_size,
        gptq_zero_point_offset=_compressed_zero_point_offset(
            model_dir,
            qc.get("config_groups") or {}),
        kv_cache_quant=_compressed_kv_cache_quant(qc),
        excluded=excluded,
    )


def _compressed_zero_point_offset(model_dir: str, config_groups: dict) -> int:
    """Return the packed-integer zero-point convention declared by the checkpoint."""
    integer_weights = [
        group.get("weights") or {} for group in config_groups.values()
        if str((group.get("weights") or {}).get("type", "")).lower() == "int"
    ]
    if not integer_weights:
        return 1

    if any(weights.get("symmetric") is False for weights in integer_weights):
        keys = _checkpoint_weight_keys(model_dir)
        if not any(key.endswith(".weight_zero_point") for key in keys):
            raise ValueError(
                "asymmetric compressed-tensors weights require explicit "
                "weight_zero_point tensors")

    # compressed-tensors stores actual zero points; it never uses GPTQ's
    # historical zero-minus-one convention.
    return 0


def _compressed_kv_cache_quant(qc: dict) -> Optional[str]:
    """Parse the compressed-tensors KV-cache quantization contract."""
    scheme = qc.get("kv_cache_scheme")
    if not scheme:
        return None
    if not isinstance(scheme, dict):
        raise ValueError(
            "compressed-tensors kv_cache_scheme must be an object")
    value_type = str(scheme.get("type", "")).lower()
    bits = int(scheme.get("num_bits") or 0)
    if value_type == "float" and bits == 8:
        return "fp8"
    raise ValueError("unsupported compressed-tensors KV-cache quantization: "
                     f"type={value_type!r}, bits={bits}")


def _parse_quant(model_dir: str,
                 config: dict,
                 submodel_prefix: str = "") -> QuantConfig:
    """Determine quantisation config from the checkpoint metadata and tensors.

    NVFP4 modules stored without a calibrated activation scale execute as
    W4A16, whatever the metadata declares.
    """
    return _weight_only_nvfp4(
        model_dir, _parse_declared_quant(model_dir, config, submodel_prefix),
        submodel_prefix)


def _weight_only_nvfp4(model_dir: str, quant: QuantConfig,
                       submodel_prefix: str) -> QuantConfig:
    """Demote NVFP4 modules without an input scale in the checkpoint to W4A16."""
    if QUANT_NVFP4 not in (quant.quant_type, *quant.layer_overrides.values()):
        return quant
    keys = set(_checkpoint_weight_keys(model_dir))
    calibrated, weight_only = set(), set()
    for suffix, input_suffix in ((".weight_scale_2", ".input_scale"),
                                 (".weight_global_scale",
                                  ".input_global_scale")):
        for key in keys:
            if not key.endswith(suffix):
                continue
            prefix = key[:-len(suffix)]
            name = _normalize_module_name(
                prefix[len(submodel_prefix):] if submodel_prefix
                and prefix.startswith(submodel_prefix) else prefix)
            (calibrated if prefix +
             input_suffix in keys else weight_only).add(name)
    if not weight_only:
        return quant
    overrides = dict(quant.layer_overrides)
    fallback = QUANT_FP16 if quant.is_mixed_precision else quant.quant_type
    for name in weight_only:
        if overrides.get(name, fallback) == QUANT_NVFP4:
            overrides[name] = QUANT_NVFP4_A16
    quant_type = quant.quant_type
    if quant_type == QUANT_NVFP4 and not calibrated:
        quant_type = QUANT_NVFP4_A16
        overrides = {
            name: (QUANT_NVFP4_A16 if value == QUANT_NVFP4 else value)
            for name, value in overrides.items()
        }
    return replace(quant, quant_type=quant_type, layer_overrides=overrides)


def _parse_declared_quant(model_dir: str,
                          config: dict,
                          submodel_prefix: str = "") -> QuantConfig:
    """Determine quantisation config from hf_quant_config.json or config.json.

    Checkpoint-provided W4A16 ``lm_head`` (``W4A16_NVFP4`` in
    ``quantized_layers``, e.g. Qwen3.6-35B-A3B-NVFP4) maps to
    :data:`QUANT_NVFP4_A16`. Excluded FP16/BF16 heads stay FP16; export does
    not invent packed NVFP4 for them.
    """
    compressed_integer = _parse_compressed_integer(model_dir, config,
                                                   submodel_prefix)
    if compressed_integer is not None:
        return compressed_integer

    grouped_nvfp4 = _parse_grouped_nvfp4(model_dir, config, submodel_prefix)
    if grouped_nvfp4 is not None:
        return grouped_nvfp4

    # ---- Sidecar hf_quant_config.json ---------------------------------------
    hf_path = os.path.join(model_dir, "hf_quant_config.json")
    if os.path.exists(hf_path):
        with open(hf_path) as f:
            hq = json.load(f)
        q = hq.get("quantization", {})
        algo = (q.get("quant_algo") or "").upper()
        if "AWQ" in algo and "W4A16" in algo:
            # Drop exclusions that the checkpoint contradicts (false positives),
            # then augment with submodules ModelOpt actually left unquantized
            # (false negatives — typically visual tower / audio encoder).
            excluded = _effective_excluded_modules(
                model_dir,
                _scope_exclusions(list(q.get("exclude_modules", [])),
                                  submodel_prefix))
            excluded.extend(m for m in _detect_modelopt_unquantized_linears(
                model_dir, submodel_prefix) if m not in excluded)
            return QuantConfig(
                quant_type=QUANT_INT4_AWQ_MODELOPT,
                group_size=int(q.get("group_size", 128)),
                excluded=excluded,
            )
        if algo == "MIXED_PRECISION":
            quantized_layers = q.get("quantized_layers", {})
            dominant, group_size, layer_overrides, layer_group_sizes = _parse_mixed_precision(
                quantized_layers,
                config.get("model_type") or "", *_moe_routing(config))
            return QuantConfig(
                quant_type=dominant,
                group_size=group_size,
                kv_cache_quant=_detect_llm_kv_cache_fp8(model_dir),
                visual_mha_quant=_detect_visual_mha_fp8(model_dir),
                excluded=_effective_excluded_modules(
                    model_dir,
                    _scope_exclusions(list(q.get("exclude_modules", [])),
                                      submodel_prefix)),
                layer_overrides=layer_overrides,
                layer_group_sizes=layer_group_sizes,
                is_mixed_precision=True,
            )
        qt = _select_nvfp4_quant_type(_algo_to_quant_type(algo))
        gs = int(q.get("group_size", 1))
        if qt == QUANT_MXFP8 and gs == 1:
            gs = 32  # MXFP8 default block_size
        excluded = _effective_excluded_modules(
            model_dir,
            _scope_exclusions(list(q.get("exclude_modules", [])),
                              submodel_prefix))
        excluded.extend(m for m in _detect_modelopt_unquantized_linears(
            model_dir, submodel_prefix) if m not in excluded)
        return QuantConfig(
            quant_type=qt,
            group_size=gs,
            kv_cache_quant=_detect_llm_kv_cache_fp8(model_dir),
            visual_mha_quant=_detect_visual_mha_fp8(model_dir),
            excluded=excluded,
        )

    # ---- Embedded quantization_config in config.json ------------------------
    qc = config.get("quantization_config")
    if qc is None:
        return QuantConfig()

    # Embedded block with ``quant_algo`` (export tool formats)
    if "quant_algo" in qc:
        algo = (qc.get("quant_algo") or "").upper()
        if algo == "MIXED_PRECISION":
            dominant, group_size, layer_overrides, layer_group_sizes = _parse_mixed_precision(
                qc.get("quantized_layers", {}),
                config.get("model_type") or "", *_moe_routing(config))
            return QuantConfig(
                quant_type=dominant,
                group_size=group_size,
                kv_cache_quant=_detect_llm_kv_cache_fp8(model_dir),
                visual_mha_quant=_detect_visual_mha_fp8(model_dir),
                excluded=_effective_excluded_modules(
                    model_dir,
                    _scope_exclusions(
                        list(qc.get("exclude_modules", qc.get("ignore", []))),
                        submodel_prefix)),
                layer_overrides=layer_overrides,
                layer_group_sizes=layer_group_sizes,
                is_mixed_precision=True,
            )
        if "W4A16" in algo and "AWQ" in algo:
            return QuantConfig(
                quant_type=QUANT_INT4_AWQ_MODELOPT,
                group_size=int(qc.get("group_size", 128)),
                excluded=_effective_excluded_modules(
                    model_dir,
                    _scope_exclusions(list(qc.get("ignore", [])),
                                      submodel_prefix)),
            )
        group_size = 1
        cg = qc.get("config_groups", {})
        if cg:
            first_group = next(iter(cg.values()), {})
            group_size = int(
                first_group.get("weights", {}).get("group_size", 1))
        elif qc.get("group_size") is not None:
            group_size = int(qc.get("group_size"))
        quant_type = _select_nvfp4_quant_type(_algo_to_quant_type(algo))
        # NVFP4 uses a fixed FP8-block group size of 16. Minimal ModelOpt
        # ``quantization_config`` blocks (``quant_algo`` only, no
        # ``config_groups`` / ``group_size`` — e.g. Qwen3-Omni Next NVFP4)
        # omit it, so default it here rather than leaving the per-tensor 1.
        if quant_type in (QUANT_NVFP4, QUANT_NVFP4_A16) and group_size == 1:
            group_size = 16
        kv = qc.get("kv_cache_scheme")
        kv_str = "fp8" if kv else None
        return QuantConfig(
            quant_type=quant_type,
            group_size=group_size,
            kv_cache_quant=kv_str,
            excluded=_effective_excluded_modules(
                model_dir,
                _scope_exclusions(list(qc.get("ignore", [])),
                                  submodel_prefix)),
        )

    # quant_method == awq (column-packed int4 checkpoints)
    if qc.get("quant_method") == "awq":
        return QuantConfig(
            quant_type=QUANT_INT4_AWQ,
            group_size=int(qc.get("group_size", 128)),
            excluded=_detect_unquantized_modules(model_dir),
        )

    # quant_method == gptq
    if qc.get("quant_method") == "gptq":
        return QuantConfig(
            quant_type=QUANT_INT4_GPTQ,
            group_size=int(qc.get("group_size", 128)),
            gptq_zero_point_offset=_detect_gptq_zero_point_offset(
                model_dir, qc),
            excluded=_detect_unquantized_modules(model_dir),
        )

    return QuantConfig()


def _detect_gptq_zero_point_offset(model_dir: str, qc: dict) -> int:
    """Infer whether GPTQ qzeros stores ``zero`` or ``zero - 1``.

    Older symmetric GPTQ checkpoints used by Qwen3 store packed ``0x77777777``
    for a real zero point of 8.  Qwen3.5 stores packed ``0x88888888`` for the
    same real zero point.  Inspecting a tiny qzeros sample lets both variants
    share the same repacking path without depending on GPTQModel/Optimum.
    """
    if not bool(qc.get("sym", False)):
        return 1

    try:
        from safetensors import safe_open

        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                weight_map: dict = json.load(f).get("weight_map", {})
            qzeros_key = next((k for k in weight_map if k.endswith(".qzeros")),
                              None)
            if qzeros_key is None:
                return 1
            shard_path = os.path.join(model_dir, weight_map[qzeros_key])
        else:
            shard_path = os.path.join(model_dir, "model.safetensors")
            if not os.path.exists(shard_path):
                return 1
            with safe_open(shard_path, framework="pt") as f:
                qzeros_key = next(
                    (k for k in f.keys() if k.endswith(".qzeros")), None)
            if qzeros_key is None:
                return 1

        with safe_open(shard_path, framework="pt") as f:
            qzeros = f.get_tensor(qzeros_key).flatten()[:1024].cpu().tolist()
    except Exception:
        return 1

    if not qzeros:
        return 1

    nibbles = []
    for value in qzeros:
        packed = int(value) & 0xFFFFFFFF
        nibbles.extend((packed >> (4 * i)) & 0xF for i in range(8))

    if nibbles and all(v == 8 for v in nibbles):
        return 0
    return 1


def _algo_to_quant_type(algo: str) -> str:
    algo = algo.upper()
    if algo == "IQ2_XXS":
        return QUANT_IQ2_XXS
    if algo == "Q8_0":
        return QUANT_Q8_0
    # MXFP8 per-block must be checked before generic FP8
    if "FP8_PB" in algo or "MXFP8" in algo:
        return QUANT_MXFP8
    if "FP8" in algo:
        return QUANT_FP8
    if "FP4" in algo or "NVFP4" in algo:
        # Weight-only NVFP4 (ModelOpt ``W4A16_NVFP4``) keeps FP16 activations.
        return QUANT_NVFP4_A16 if "A16" in algo else QUANT_NVFP4
    # W4A16_AWQ from ModelOpt unified checkpoints uses prepacked uint8 weights;
    # plain AWQ / INT4_AWQ from HuggingFace uses column-packed int32 qweight.
    if "W4A16" in algo and "AWQ" in algo:
        return QUANT_INT4_AWQ_MODELOPT
    if "AWQ" in algo or "INT4_AWQ" in algo:
        return QUANT_INT4_AWQ
    if "W8A8" in algo or "INT8" in algo:
        return QUANT_INT8_SQ
    return QUANT_FP16


def _select_nvfp4_quant_type(quant_type: str) -> str:
    """Resolve packed NVFP4 weights to the requested activation precision."""
    if quant_type == QUANT_NVFP4 and not _DEFAULT_QUANTIZE_ACTIVATIONS:
        return QUANT_NVFP4_A16
    return quant_type


# Fp16MoePlugin routes only these expert counts and at most this top-k.
_FP16_MOE_EXPERT_COUNTS = (128, 256)
_FP16_MOE_MAX_TOP_K = 8
_MOE_EXPERT_COUNT_KEYS = ("num_experts", "num_local_experts",
                          "n_routed_experts", "moe_num_experts")


def _fp16_moe_routes(num_experts: int, top_k: int) -> bool:
    return (num_experts in _FP16_MOE_EXPERT_COUNTS
            and 0 < top_k <= _FP16_MOE_MAX_TOP_K)


def _moe_routing(config: dict) -> "tuple[int, int]":
    """``(num_experts, top_k)`` of an HF config dict, zeros for a dense model."""
    count = next((int(config[key])
                  for key in _MOE_EXPERT_COUNT_KEYS if config.get(key)), 0)
    return count, int(config.get("num_experts_per_tok") or 0)


def _parse_mixed_precision(quantized_layers: dict,
                           model_type: str = "",
                           num_experts: int = 0,
                           top_k: int = 0) -> "tuple[str, int, dict, dict]":
    """Parse MIXED_PRECISION quantized_layers dict.

    Returns ``(dominant_quant_type, dominant_group_size, layer_overrides,
    layer_group_sizes)``.
    ``layer_overrides`` maps **every** quantized module name to its quant-type
    string.  Modules not listed in ``quantized_layers`` are unquantized (FP16);
    ``make_linear`` falls back to FP16 when a module_name is absent from
    ``layer_overrides``.

    Block-scaled FP8 (``FP8_BLOCK_SCALES``) has no form in the graph; the loader
    dequantizes those weights, so a dense module runs FP16 and an expert bank
    runs FP16 when ``Fp16MoePlugin`` can route it, else requantized W4A16.
    """
    from collections import Counter

    def _is_fp8_block(algo: str) -> bool:
        return "FP8_BLOCK" in algo

    def _mixed_quant_type(algo: str) -> str:
        if _is_fp8_block(algo):
            return QUANT_FP16
        return _select_nvfp4_quant_type(_algo_to_quant_type(algo))

    def _group_size(layer_config: dict, quant_type: str) -> int:
        configured = int(layer_config.get("group_size", 1))
        if configured != 1:
            return configured
        if quant_type == QUANT_MXFP8:
            return 32
        if quant_type in (QUANT_NVFP4, QUANT_NVFP4_A16):
            return 16
        return configured

    algo_count: Counter = Counter()
    algo_group_size: dict = {}
    for layer_cfg in quantized_layers.values():
        algo = layer_cfg.get("quant_algo", "").upper()
        algo_count[algo] += 1
        if algo not in algo_group_size:
            algo_group_size[algo] = _group_size(layer_cfg,
                                                _mixed_quant_type(algo))
    if not algo_count:
        return QUANT_FP16, 1, {}, {}

    dominant_algo = algo_count.most_common(1)[0][0]
    dominant_type = _mixed_quant_type(dominant_algo)
    dominant_group_size = algo_group_size.get(dominant_algo, 1)
    # Expand fused projection keys (``self_attn.qkv_proj``,
    # ``mlp.gate_up_proj``) into the split names ``make_linear`` looks up
    # (``q_proj``/``k_proj``/``v_proj`` and ``gate_proj``/``up_proj``).
    layer_overrides: dict = {}
    layer_group_sizes: dict = {}
    for name, layer_cfg in quantized_layers.items():
        algo = layer_cfg.get("quant_algo", "").upper()
        short_name = _normalize_module_name(name)
        quant_type = _mixed_quant_type(algo)
        group_size = _group_size(layer_cfg, quant_type)
        if (_is_muse_glimmer_model_type(model_type)
                and short_name.endswith(".self_attn.output_gate_proj")):
            short_name = short_name[:-len(".output_gate_proj")] + ".gate_proj"
        if short_name.endswith(".self_attn.qkv_proj"):
            prefix = short_name[:-len("qkv_proj")]
            module_names = tuple(f"{prefix}{proj}"
                                 for proj in ("q_proj", "k_proj", "v_proj"))
        elif short_name.endswith(".mlp.gate_up_proj"):
            prefix = short_name[:-len("gate_up_proj")]
            module_names = tuple(f"{prefix}{proj}"
                                 for proj in ("gate_proj", "up_proj"))
        else:
            module_names = (short_name, )
        if _is_fp8_block(algo):
            if (short_name.endswith(".mlp.experts")
                    and not _fp16_moe_routes(num_experts, top_k)):
                quant_type, group_size = QUANT_NVFP4_A16, 16
            else:
                quant_type, group_size = QUANT_FP16, 1
        for module_name in module_names:
            layer_overrides[module_name] = quant_type
            layer_group_sizes[module_name] = group_size
    return (dominant_type, dominant_group_size, layer_overrides,
            layer_group_sizes)


def _kv_norm(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return s.strip().lower() or None


def _safetensor_keys(model_dir: str) -> Optional[List[str]]:
    """Return all tensor keys from sharded or single-file safetensors."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return list(json.load(f).get("weight_map", {}).keys())
    st_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(st_path):
        return None
    from safetensors import safe_open
    with safe_open(st_path, framework="pt") as f:
        return list(f.keys())


def _detect_visual_mha_fp8(model_dir: str) -> Optional[str]:
    """Return ``"fp8"`` iff the checkpoint carries visual-MHA Q/K/V scales.

    Self-describing detection: the presence of ``<visual_prefix>*.q_scale``
    buffers in the safetensors is the ground-truth signal that ViT FP8
    MHA was calibrated. Avoids trusting modelopt's hf_quant_config.json
    metadata (which has no ``visual_mha_quant_algo`` slot and conflates
    ViT q/k/v_bmm with the LLM KV cache).
    """
    keys = _safetensor_keys(model_dir)
    if keys is None:
        return None
    for k in keys:
        if k.endswith(".q_scale") and any(h in k for h in VISUAL_PREFIXES):
            return "fp8"
    return None


def _detect_llm_kv_cache_fp8(model_dir: str) -> Optional[str]:
    """Return ``"fp8"`` iff the checkpoint carries LLM KV-cache K/V scales.

    Self-describing detection: an ``<llm_attn>.k_proj.k_scale`` buffer
    outside of any visual / vision subtree is the ground-truth signal.
    Modelopt's ``hf_quant_config.json:kv_cache_quant_algo`` field would
    say "FP8" even when only ViT MHA was requested (because its writer
    trips on any enabled ``k_bmm_quantizer``, including the visual one);
    ignoring that field and looking at the checkpoint directly avoids
    the false positive.
    """
    keys = _safetensor_keys(model_dir)
    if keys is None:
        return None
    for k in keys:
        if k.endswith(".k_proj.k_scale") and not any(h in k
                                                     for h in VISUAL_PREFIXES):
            return "fp8"
    return None
