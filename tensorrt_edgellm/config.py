# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    int4_awq            - AWQ INT4 group quantization (column-packed int32 checkpoints)
    int4_awq_modelopt   - W4A16_AWQ pre-packed uint8 ``[out//2, in]`` checkpoints
    int4_gptq           - GPTQ INT4 group quantization
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

import json
import os
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import torch

from .checkpoint.checkpoint_utils import load_checkpoint_config_dicts

# ---------------------------------------------------------------------------
# Quantization type constants
# ---------------------------------------------------------------------------

QUANT_FP16 = "fp16"
QUANT_FP8 = "fp8"
QUANT_MXFP8 = "mxfp8"
QUANT_NVFP4 = "nvfp4"
QUANT_INT4_AWQ = "int4_awq"
QUANT_INT4_AWQ_MODELOPT = "int4_awq_modelopt"
QUANT_INT4_GPTQ = "int4_gptq"
QUANT_INT8_SQ = "int8_sq"
# ``quant_algo`` value in hf_quant_config for MIXED_PRECISION only.
# After parsing, :attr:`QuantConfig.quant_type` is always a concrete type
# (dominant algo); per-layer differences live in ``layer_overrides``.
QUANT_MIXED = "mixed_precision"

# Deployment-target backend for NVFP4 MoE plugins. Even though both backends
# read the same NVFP4 checkpoint, they emit *different* ONNX nodes
# (``Nvfp4MoePlugin`` vs ``NvFP4MoEPluginGeforce``) with different input
# contracts and weight layouts, so the backend must be picked at export
# time. Surfaced through ``QuantConfig`` for convenience (it travels with
# the rest of the export-time config); override via the checkpoint's
# ``hf_quant_config.json`` / ``quantization_config`` block (key
# ``nvfp4_moe_backend``) or by mutating ``config.quant.nvfp4_moe_backend``
# before construction. Missing keys target the Thor plugin by default.
NVFP4_MOE_BACKEND_THOR = "thor"  # CuTeDSL Nvfp4MoePlugin (SM100/101/110)
NVFP4_MOE_BACKEND_GEFORCE = "geforce"  # NvFP4MoEPluginGeforce (SM120/121)
_VALID_NVFP4_MOE_BACKENDS = (NVFP4_MOE_BACKEND_THOR, NVFP4_MOE_BACKEND_GEFORCE)

# Default RoPE base frequency (used when config omits rope_theta)
_DEFAULT_ROPE_THETA = 10000.0

# Layer-type labels
LAYER_ATTN = "attention"
LAYER_MAMBA = "mamba"
LAYER_MLP = "mlp"
LAYER_GDN = "gdn"  # GatedDeltaNet linear attention (Qwen3.5)
LAYER_MOE = "moe"


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
        if isinstance(nested, dict) and nested.get("rope_theta") is not None:
            return float(nested["rope_theta"])
    return _DEFAULT_ROPE_THETA


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
    # module names excluded from quantisation (typically ["lm_head"])
    excluded: List[str] = field(default_factory=list)
    # Per-layer quant type overrides for MIXED_PRECISION checkpoints.
    # Maps a module name (e.g. "lm_head") to a quant-type string.
    # make_linear() uses module_name together with ``excluded`` and (for
    # lm_head) ``ModelConfig.tie_word_embeddings`` to pick FP16 vs overrides.
    layer_overrides: dict = field(default_factory=dict)
    # True when quant_algo is MIXED_PRECISION: unlisted modules are FP16.
    is_mixed_precision: bool = False
    # Deployment-target backend for NVFP4 MoE plugins (only consumed when
    # ``quant_type == QUANT_NVFP4`` *and* the model is an MoE arch). See the
    # ``NVFP4_MOE_BACKEND_*`` constants above for the supported values.
    # Default targets Thor SM110.
    nvfp4_moe_backend: str = NVFP4_MOE_BACKEND_THOR

    def __post_init__(self) -> None:
        if self.nvfp4_moe_backend not in _VALID_NVFP4_MOE_BACKENDS:
            raise ValueError(
                f"QuantConfig.nvfp4_moe_backend must be one of "
                f"{_VALID_NVFP4_MOE_BACKENDS}, got {self.nvfp4_moe_backend!r}")

    @property
    def is_quantized(self) -> bool:
        return self.quant_type != QUANT_FP16

    @property
    def uses_nvfp4_weights(self) -> bool:
        """True if any linear uses NVFP4 weights (dominant quant or layer override)."""
        if self.quant_type == QUANT_NVFP4:
            return True
        return any(v == QUANT_NVFP4 for v in self.layer_overrides.values())

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
    if module_name and module_name in quant.excluded:
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
class Gemma4Config:
    """Gemma 4 (``gemma4_text``) architecture parameters.

    Gemma 4 interleaves local *sliding-window* attention with global *full*
    attention, and the two layer types differ in more than the attention mask:
    they use different head dimensions (``head_dim`` vs ``global_head_dim``) and
    different RoPE parameters (a plain ``rope_theta`` for local layers, a
    proportional RoPE with its own ``rope_theta`` and ``partial_rotary_factor``
    for global layers).  Gemma 4 also adds Per-Layer Embeddings (PLE), key/value
    sharing across the trailing decoder layers, and final-logit soft-capping.

    None of these fit the flat :class:`ModelConfig` shared by Llama-style dense
    models, so they are collected here and attached as
    :attr:`ModelConfig.gemma4_cfg`.  See ``models/gemma4`` for the consuming
    modeling code.
    """

    #: Per-layer attention type: ``True`` for global (full) attention, ``False``
    #: for local (sliding-window) attention.  Length == ``num_hidden_layers``.
    is_global_layer: List[bool]
    #: Head dimension for local (sliding) attention layers (``head_dim``).
    local_head_dim: int
    #: Head dimension for global (full) attention layers (``global_head_dim``).
    global_head_dim: int
    #: Sliding-window size used by the local attention layers.
    sliding_window: int
    #: Number of trailing decoder layers that reuse a previous layer's KV
    #: projections (``num_kv_shared_layers``); 0 means no sharing.
    num_kv_shared_layers: int
    #: KV heads for global layers (``num_global_key_value_heads``); falls back to
    #: ``num_key_value_heads`` when the checkpoint leaves it unset.
    num_global_key_value_heads: int
    #: When ``True``, global layers reuse the key projection as the value
    #: projection (``attention_k_eq_v``).
    attention_k_eq_v: bool
    #: RoPE base frequency for local (sliding) layers.
    sliding_rope_theta: float
    #: RoPE base frequency for global (full) layers.
    global_rope_theta: float
    #: Fraction of ``global_head_dim`` that receives RoPE on global layers
    #: (Proportional RoPE; the global rope params' ``partial_rotary_factor``).
    global_partial_rotary_factor: float
    #: RoPE variant for global layers (e.g. ``"proportional"``).
    global_rope_type: str
    #: Per-Layer-Embedding hidden width (``hidden_size_per_layer_input``); 0
    #: disables PLE.
    hidden_size_per_layer_input: int
    #: Vocabulary size of the PLE table (``vocab_size_per_layer_input``).
    vocab_size_per_layer_input: int
    #: Final-logit soft-capping value (``final_logit_softcapping``); ``None`` when
    #: the checkpoint does not soft-cap logits.
    final_logit_softcapping: Optional[float]
    #: When ``True``, the KV-shared layers use a double-width MLP intermediate
    #: size (``use_double_wide_mlp``).
    use_double_wide_mlp: bool
    #: Feed-forward activation name (e.g. ``"gelu_pytorch_tanh"``).
    hidden_activation: str

    def first_kv_shared_layer_idx(self, num_hidden_layers: int) -> int:
        """Index of the first KV-sharing layer (== ``num_hidden_layers`` if none)."""
        return num_hidden_layers - self.num_kv_shared_layers

    @property
    def num_global_layers(self) -> int:
        """Count of global (full) attention layers."""
        return sum(1 for is_global in self.is_global_layer if is_global)

    @property
    def num_local_layers(self) -> int:
        """Count of local (sliding-window) attention layers."""
        return sum(1 for is_global in self.is_global_layer if not is_global)

    @property
    def uses_per_layer_embeddings(self) -> bool:
        """True when the model carries a Per-Layer Embedding pathway."""
        return self.hidden_size_per_layer_input > 0


@dataclass
class ModelConfig:
    """Flat model hyper-parameter config consumed by module builders."""

    # ------------------------------------------------------------------ arch
    model_type: str  # e.g. "qwen3", "llama", "hybrid_mamba"
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
    # RoPE scaling config (e.g. {"type": "dynamic", "factor": 2.0} for Qwen2).
    # None means no scaling (standard RoPE).
    rope_scaling: Optional[dict] = None
    # For longrope: original context window size before extension.
    original_max_position_embeddings: Optional[int] = None
    # Fraction of head_dim used for RoPE (e.g. 0.75 for phi3/phi4, 1.0 for most others).
    partial_rotary_factor: float = 1.0
    # ------------------------------------------ model-family feature flags
    # Per-head RMSNorm after Q and K projections.
    # Auto-detected from checkpoint key names; not inferred from model_type.
    has_qk_norm: bool = False
    # Bias on q/k/v projections.  Read from config.json "attention_bias".
    attention_bias: bool = False
    # Weight dtype in the checkpoint
    torch_dtype: str = "bfloat16"
    # When True, embed_tokens and lm_head share the same weight tensor
    tie_word_embeddings: bool = False
    # Sliding window attention size; -1 means no sliding window.
    sliding_window_size: int = -1
    # ------------------------------------------ per-layer block types
    # One entry per hidden layer: LAYER_ATTN, LAYER_MAMBA, LAYER_MLP, or LAYER_MOE.
    layer_types: List[str] = field(default_factory=list)
    # ------------------------------------------ multimodal deepstack (VL)
    # Number of deepstack visual embedding tensors injected into the first N
    # hidden layers. Prefer ``vision_config.deepstack_visual_indexes`` length on
    # the root config when present; else ``num_deepstack_features`` or fallback.
    num_deepstack_features: int = 0
    # -------------------------------------------------- quantization config
    quant: QuantConfig = field(default_factory=QuantConfig)
    # ------------------------------------------ mamba / hybrid config
    mamba_cfg: Optional[MambaConfig] = None
    # ------------------------------------------ gdn / hybrid config
    gdn_cfg: Optional[GdnConfig] = None
    # ------------------------------------------ gemma4 (gemma4_text) config
    # Populated only for Gemma 4 checkpoints; carries the per-layer-type
    # attention, Per-Layer-Embedding, KV-sharing, and soft-capping parameters
    # that do not fit the flat dense-model fields above.
    gemma4_cfg: Optional[Gemma4Config] = None
    # ------------------------------------------ gated attention (Qwen3.5)
    attn_output_gate: bool = False
    # ------------------------------------------ MTP config
    mtp_num_hidden_layers: Optional[int] = None
    mtp_use_dedicated_embeddings: bool = False
    # When True, the standard CausalLM is exported as the MTP base model variant
    # with tree-attention inputs (attention_mask, attention_pos_id) and
    # an extra hidden_states output.
    mtp_base: bool = False
    # ------------------------------------------ EAGLE3 draft config
    draft_vocab_size: Optional[int] = None
    target_hidden_size: Optional[int] = None
    # ------------------------------------------ EAGLE3 base config
    # When True, the standard CausalLM is exported as an EAGLE3 base model
    # with tree-attention inputs (attention_mask, attention_pos_id) and
    # an extra hidden_states output (concatenated from 3 selected layers).
    eagle_base: bool = False
    # ------------------------------------------ sparse MoE config (Qwen3-style)
    # num_experts=0 means dense (no MoE).
    num_experts: int = 0
    n_routed_experts: int = 0
    num_experts_per_tok: int = 0
    # Expert MLP intermediate size (may differ from dense intermediate_size).
    moe_intermediate_size: int = 0
    moe_shared_expert_intermediate_size: int = 0
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

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def is_eagle3_draft(self) -> bool:
        return self.draft_vocab_size is not None

    @property
    def is_mtp_draft(self) -> bool:
        """True for a derived MTP draft config built from a base checkpoint."""
        return bool(self.mtp_num_hidden_layers is not None
                    and self.gdn_cfg is None and not self.mtp_base
                    and not self.is_eagle3_draft)

    @property
    def eagle3_target_hidden_size(self) -> int:
        return self.target_hidden_size or self.hidden_size

    @property
    def is_hybrid(self) -> bool:
        return self.mamba_cfg is not None or self.gdn_cfg is not None

    @property
    def is_gemma4(self) -> bool:
        return self.gemma4_cfg is not None

    @property
    def is_nemotron_h(self) -> bool:
        return (self.model_type or "").lower().startswith("nemotron_h")

    @property
    def num_attn_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == LAYER_ATTN)

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
    def compute_dtype(self) -> "torch.dtype":  # noqa: F821
        import torch
        _MAP = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return _MAP.get(self.torch_dtype, torch.bfloat16)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "ModelConfig":
        """Load a ModelConfig from a checkpoint directory.

        Loads architecture hyper-parameters via ``AutoConfig`` (see
        :func:`checkpoint_utils.load_checkpoint_config_dicts`) and then
        either ``hf_quant_config.json`` or the embedded ``quantization_config``
        block to determine
        the quantisation scheme.

        ``has_qk_norm`` is auto-detected by scanning the safetensors key index
        for ``.q_norm.weight`` entries - no model-type assumptions are made.
        """
        root, llm_dict = load_checkpoint_config_dicts(model_dir)

        model_type = llm_dict.get("model_type", "llama")
        hidden_size = llm_dict["hidden_size"]
        num_attn_heads = llm_dict["num_attention_heads"]
        head_dim = llm_dict.get("head_dim", hidden_size // num_attn_heads)

        quant = _parse_quant(model_dir, llm_dict)
        layer_types = _parse_layer_types(llm_dict)
        mamba_cfg = _parse_mamba_cfg(llm_dict,
                                     layer_types,
                                     model_dir=model_dir)
        gdn_cfg = _parse_gdn_cfg(llm_dict, layer_types)
        gemma4_cfg = _parse_gemma4_cfg(llm_dict)
        has_qk_norm = _detect_has_qk_norm(model_dir)

        # MTP config
        mtp_num_hidden_layers = llm_dict.get("mtp_num_hidden_layers")
        if mtp_num_hidden_layers is not None:
            mtp_num_hidden_layers = int(mtp_num_hidden_layers)
        mtp_use_dedicated_embeddings = bool(
            llm_dict.get("mtp_use_dedicated_embeddings", False))
        _validate_mtp_constraints(
            model_type=model_type,
            mtp_num_hidden_layers=mtp_num_hidden_layers,
            mtp_use_dedicated_embeddings=mtp_use_dedicated_embeddings,
        )

        # EAGLE3 draft model fields
        draft_vocab_size = llm_dict.get("draft_vocab_size", None)
        target_hidden_size = llm_dict.get("target_hidden_size", None)
        # Sliding window: only active when use_sliding_window=True.
        use_sw = llm_dict.get("use_sliding_window", False)
        sw_raw = llm_dict.get("sliding_window") if use_sw else None
        sliding_window_size = int(sw_raw) if sw_raw is not None else -1

        # Sparse MoE fields.  HF uses "num_local_experts" as the internal key
        # and maps "num_experts" → "num_local_experts" via attribute_map.
        num_experts = int(
            llm_dict.get("num_experts", llm_dict.get("num_local_experts", 0))
            or 0)
        # ``or 0`` (rather than a get-default) so configs that explicitly set
        # these optional keys to ``null`` — e.g. dense Gemma 4 checkpoints with
        # ``"moe_intermediate_size": null`` — parse as 0 instead of crashing.
        num_experts_per_tok = int(llm_dict.get("num_experts_per_tok") or 0)
        moe_intermediate_size = int(llm_dict.get("moe_intermediate_size") or 0)
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

        intermediate_size = int(
            llm_dict.get("intermediate_size")
            or llm_dict.get("shared_expert_intermediate_size")
            or llm_dict.get("moe_shared_expert_intermediate_size")
            or llm_dict.get("moe_intermediate_size", 0))

        return cls(
            model_type=model_type,
            hidden_size=hidden_size,
            num_hidden_layers=llm_dict["num_hidden_layers"],
            num_attention_heads=num_attn_heads,
            num_key_value_heads=llm_dict.get("num_key_value_heads",
                                             num_attn_heads),
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            rms_norm_eps=llm_dict.get("rms_norm_eps", 1e-6),
            vocab_size=llm_dict["vocab_size"],
            rope_theta=_get_rope_theta(llm_dict),
            max_position_embeddings=llm_dict.get("max_position_embeddings",
                                                 4096),
            rope_scaling=(llm_dict.get("rope_scaling")
                          or llm_dict.get("rope_parameters") or None),
            original_max_position_embeddings=llm_dict.get(
                "original_max_position_embeddings", None),
            partial_rotary_factor=_get_partial_rotary_factor(llm_dict),
            has_qk_norm=has_qk_norm,
            attention_bias=bool(llm_dict.get("attention_bias", False)),
            torch_dtype=llm_dict.get("torch_dtype",
                                     llm_dict.get("dtype", "bfloat16")),
            tie_word_embeddings=llm_dict.get("tie_word_embeddings", False),
            sliding_window_size=sliding_window_size,
            layer_types=layer_types,
            quant=quant,
            mamba_cfg=mamba_cfg,
            gdn_cfg=gdn_cfg,
            gemma4_cfg=gemma4_cfg,
            attn_output_gate=bool(llm_dict.get("attn_output_gate", False)),
            mtp_num_hidden_layers=mtp_num_hidden_layers,
            mtp_use_dedicated_embeddings=mtp_use_dedicated_embeddings,
            mtp_base=bool(llm_dict.get("mtp_base", False)),
            num_deepstack_features=_parse_num_deepstack_features(
                llm_dict, model_type, root_config=root),
            draft_vocab_size=draft_vocab_size,
            target_hidden_size=target_hidden_size,
            num_experts=num_experts,
            n_routed_experts=llm_dict.get("n_routed_experts", 0),
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            moe_shared_expert_intermediate_size=
            moe_shared_expert_intermediate_size,
            routed_scaling_factor=routed_scaling_factor,
            n_group=n_group,
            topk_group=topk_group,
            decoder_sparse_step=decoder_sparse_step,
            mlp_only_layers=mlp_only_layers,
            norm_topk_prob=norm_topk_prob,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# When HF omits ``deepstack_visual_indexes`` / ``num_deepstack_features``,
# these model families still expect three visual deepstack injections.
_DEEPSTACK_MODEL_TYPES = ("qwen3_vl", "qwen3_omni")


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

    # MTP modules in the exclude list → unquantized (FP16); otherwise inherit
    # base quant.  Only *compute linear* modules matter (fc, proj, etc.).
    # Norms, embeddings, and lm_head always appear in `excluded` when lm_head
    # is FP16 — their presence does NOT mean the whole draft is unquantized.
    _MTP_COMPUTE_PREFIXES = ("mtp.fc", "mtp.layers.")
    mtp_is_quantized = not any(
        any(e.startswith(p) for p in _MTP_COMPUTE_PREFIXES) and
        ("norm" not in e and "embed" not in e)
        for e in base_config.quant.excluded)

    if mtp_is_quantized:
        # MTP draft modules are independently quantized.  Strip base-model
        # layer_overrides that use module paths absent from the draft
        # (e.g. "model.layers.0.linear_attn.in_proj_qkv") — they would
        # cause spurious FP16 fallback for draft-specific layers like "fc".
        # Keep entries whose keys also exist in the draft module namespace
        # (e.g. "lm_head") so that lm_head_quantization is honoured.
        _DRAFT_MODULE_PREFIXES = ("lm_head", "fc", "layers.", "norm")
        draft_overrides = {
            k: v
            for k, v in base_config.quant.layer_overrides.items()
            if any(k == p or k.startswith(p) for p in _DRAFT_MODULE_PREFIXES)
        }
        # Preserve MTP-specific exclusions (e.g. mtp.lm_head when lm_head
        # is FP16) but drop base-model exclusions irrelevant to the draft.
        draft_excluded = [
            e[len("mtp."):] for e in base_config.quant.excluded
            if e.startswith("mtp.")
        ]
        # is_mixed_precision=False so unlisted modules (fc, q_proj, etc.)
        # fall back to the dominant quant type, not FP16.  Explicit
        # overrides (lm_head→fp8) still take effect via layer_overrides.
        draft_quant = replace(base_config.quant,
                              excluded=draft_excluded,
                              layer_overrides=draft_overrides,
                              is_mixed_precision=False)
    else:
        draft_quant = QuantConfig()

    return replace(
        base_config,
        num_hidden_layers=mtp_num_hidden_layers,
        layer_types=[LAYER_ATTN] * mtp_num_hidden_layers,
        gdn_cfg=None,
        mtp_base=False,
        quant=draft_quant,
        tie_word_embeddings=False,
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
    merged = f"{model_type} {root_mt}"
    if any(t in merged for t in _DEEPSTACK_MODEL_TYPES):
        return 3
    return 0


def _validate_mtp_constraints(
    *,
    model_type: str,
    mtp_num_hidden_layers: Optional[int],
    mtp_use_dedicated_embeddings: bool,
) -> None:
    """Validate the currently supported MTP config subset."""
    if mtp_num_hidden_layers is None and not mtp_use_dedicated_embeddings:
        return
    if model_type not in ("qwen3_5_text", "qwen3_5_moe_text"):
        raise NotImplementedError(
            "MTP config parsing is only supported for Qwen3.5 checkpoints.")
    if mtp_num_hidden_layers != 1:
        raise NotImplementedError(
            "Only mtp_num_hidden_layers == 1 is supported for Qwen3.5 dense MTP."
        )
    if mtp_use_dedicated_embeddings:
        raise NotImplementedError(
            "Dedicated MTP embeddings are not supported for Qwen3.5 dense MTP."
        )


def _parse_layer_types(config: dict) -> List[str]:
    """Return per-layer block type list from config.

    Reads ``layers_block_type`` or ``layer_types`` directly if present.
    For models using ``hybrid_override_pattern`` (e.g. NemotronH), parses the
    pattern string where ``M`` = mamba, ``-`` = mlp, ``*`` = attention.
    Falls back to all attention layers.

    Qwen3.5 uses ``layer_types`` with values ``"linear_attention"`` (GDN)
    and ``"full_attention"``.
    """
    raw = config.get("layers_block_type") or config.get("layer_types")
    if raw is not None:
        result = []
        for bt in raw:
            bt_lower = str(bt).lower()
            if bt_lower == "linear_attention":
                result.append(LAYER_GDN)
            elif "mamba" in bt_lower:
                result.append(LAYER_MAMBA)
            elif bt_lower == "moe":
                result.append(LAYER_MOE)
            elif "mlp" in bt_lower:
                result.append(LAYER_MLP)
            else:
                result.append(LAYER_ATTN)
        return result
    pattern = config.get("hybrid_override_pattern")
    if pattern is not None:
        _PATTERN_MAP = {
            "M": LAYER_MAMBA,
            "-": LAYER_MLP,
            "*": LAYER_ATTN,
            "E": LAYER_MOE,
        }
        return [_PATTERN_MAP[ch] for ch in pattern if ch in _PATTERN_MAP]
    n = config["num_hidden_layers"]
    return [LAYER_ATTN] * n


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
    )


# Default RoPE base for Gemma 4 global (full) attention layers when the
# checkpoint's ``rope_parameters`` omits it.
_GEMMA4_DEFAULT_GLOBAL_ROPE_THETA = 1_000_000.0


def _parse_gemma4_cfg(llm_dict: Dict[str, Any]) -> Optional[Gemma4Config]:
    """Return a :class:`Gemma4Config` for ``gemma4_text`` checkpoints, else None.

    Gemma 4 stores ``rope_parameters`` as a mapping keyed by layer type
    (``"sliding_attention"`` / ``"full_attention"``) rather than a single flat
    block, and ``layer_types`` marks each decoder layer as sliding (local) or
    full (global).  Both are parsed here, along with the per-layer-type head
    dimensions, KV-sharing, Per-Layer-Embedding, and soft-capping fields.

    Returns ``None`` for non-Gemma-4 checkpoints so the parser is inert for
    every other model family.
    """
    model_type = str(llm_dict.get("model_type", "")).lower()
    if not model_type.startswith("gemma4"):
        return None

    raw_layer_types = llm_dict.get("layer_types") or []
    is_global_layer = [
        str(layer_type).lower() == "full_attention"
        for layer_type in raw_layer_types
    ]

    rope_params = llm_dict.get("rope_parameters")
    if not isinstance(rope_params, dict):
        rope_params = {}
    sliding_rope = rope_params.get("sliding_attention") or {}
    global_rope = rope_params.get("full_attention") or {}

    num_key_value_heads = int(llm_dict.get("num_key_value_heads", 1))
    raw_global_kv_heads = llm_dict.get("num_global_key_value_heads")
    num_global_key_value_heads = num_key_value_heads
    if raw_global_kv_heads is not None:
        num_global_key_value_heads = int(raw_global_kv_heads)

    hidden_size = int(llm_dict["hidden_size"])
    num_attention_heads = int(llm_dict["num_attention_heads"])
    local_head_dim = int(
        llm_dict.get("head_dim", hidden_size // num_attention_heads))
    global_head_dim = int(llm_dict.get("global_head_dim") or local_head_dim)

    softcap = llm_dict.get("final_logit_softcapping")
    final_logit_softcapping = float(softcap) if softcap is not None else None

    return Gemma4Config(
        is_global_layer=is_global_layer,
        local_head_dim=local_head_dim,
        global_head_dim=global_head_dim,
        sliding_window=int(llm_dict.get("sliding_window") or 0),
        num_kv_shared_layers=int(llm_dict.get("num_kv_shared_layers") or 0),
        num_global_key_value_heads=num_global_key_value_heads,
        attention_k_eq_v=bool(llm_dict.get("attention_k_eq_v", False)),
        sliding_rope_theta=float(
            sliding_rope.get("rope_theta", _DEFAULT_ROPE_THETA)),
        global_rope_theta=float(
            global_rope.get("rope_theta", _GEMMA4_DEFAULT_GLOBAL_ROPE_THETA)),
        global_partial_rotary_factor=float(
            global_rope.get("partial_rotary_factor", 1.0)),
        global_rope_type=str(global_rope.get("rope_type", "default")),
        hidden_size_per_layer_input=int(
            llm_dict.get("hidden_size_per_layer_input") or 0),
        vocab_size_per_layer_input=int(
            llm_dict.get("vocab_size_per_layer_input")
            or llm_dict.get("vocab_size") or 0),
        final_logit_softcapping=final_logit_softcapping,
        use_double_wide_mlp=bool(llm_dict.get("use_double_wide_mlp", False)),
        hidden_activation=str(
            llm_dict.get("hidden_activation", "gelu_pytorch_tanh")),
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
        if isinstance(
                nested,
                dict) and nested.get("partial_rotary_factor") is not None:
            return float(nested["partial_rotary_factor"])
    return 1.0


def _detect_has_qk_norm(model_dir: str) -> bool:
    """Detect QK-norm by scanning checkpoint key names for ``.q_norm.weight``.

    This is model-agnostic: any architecture that stores per-head Q/K norms
    as ``*.q_norm.weight`` buffers will be detected correctly.
    """
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        return any(".q_norm.weight" in k
                   for k in index.get("weight_map", {}).keys())

    # Single-shard: scan keys without loading tensors
    single_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single_path):
        try:
            from safetensors import safe_open
            with safe_open(single_path, framework="pt") as f:
                return any(".q_norm.weight" in k for k in f.keys())
        except (OSError, ImportError):
            pass
    return False


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
    for prefix in _VL_LLM_PREFIXES + ("model.", ):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _detect_unquantized_modules(model_dir: str) -> List[str]:
    """Return module names whose weights are plain float (not int4 quantized).

    Some layers (often ``lm_head``) use ``*.weight`` instead of ``*.qweight``.
    VL wrapper prefixes (``language_model.`` etc.) are stripped so that the
    returned names match the short names used by ``make_linear()``.
    """
    all_keys: List[str] = []
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        all_keys = list(index.get("weight_map", {}).keys())
    else:
        single_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(single_path):
            try:
                from safetensors import safe_open
                with safe_open(single_path, framework="pt") as f:
                    all_keys = list(f.keys())
            except (OSError, ImportError):
                pass

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
        _strip_vl_prefix(m) for m in (weight_modules - qweight_modules)
    ]
    # lm_head may have neither .weight nor .qweight when tie_word_embeddings=True
    # (the checkpoint omits lm_head.weight entirely).  Treat it as FP16 so that
    # tie_weights() can clone embed_tokens.weight into it after loading.
    all_linear_stripped = {
        _strip_vl_prefix(m)
        for m in qweight_modules | weight_modules
    }
    if "lm_head" not in all_linear_stripped:
        excluded.append("lm_head")
    return _with_gdn_fused_exclusions(excluded)


_GDN_INPUT_PROJ_MODULES = ("in_proj_qkv", "in_proj_z", "in_proj_b",
                           "in_proj_a")


def _with_gdn_fused_exclusions(modules: List[str]) -> List[str]:
    """Add synthetic GDN fused projections when all source projections are FP16."""
    result = set(modules)
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
    all_keys: List[str] = []
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        all_keys = list(index.get("weight_map", {}).keys())
    else:
        single_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(single_path):
            try:
                from safetensors import safe_open
                with safe_open(single_path, framework="pt") as f:
                    all_keys = list(f.keys())
            except (OSError, ImportError):
                pass

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


def _detect_modelopt_unquantized_linears(model_dir: str) -> List[str]:
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
    all_keys: List[str] = []
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path) as f:
                index = json.load(f)
            all_keys = list(index.get("weight_map", {}).keys())
        except (OSError, json.JSONDecodeError):
            pass
    if not all_keys:
        single_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(single_path):
            try:
                from safetensors import safe_open
                with safe_open(single_path, framework="pt") as f:
                    all_keys = list(f.keys())
            except (OSError, ImportError):
                pass

    if not all_keys:
        return []

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


def _parse_nvfp4_moe_backend(blob: dict) -> str:
    """Read ``nvfp4_moe_backend`` from a quantization-config blob.

    Accepts ``thor`` / ``geforce`` (case-insensitive). Missing key falls back
    to :data:`NVFP4_MOE_BACKEND_THOR`. Raises on an unknown explicit value so
    typos don't silently degrade the deployment target.
    """
    raw = blob.get("nvfp4_moe_backend")
    if raw is None:
        return NVFP4_MOE_BACKEND_THOR
    backend = str(raw).strip().lower()
    if backend not in _VALID_NVFP4_MOE_BACKENDS:
        raise ValueError(
            f"nvfp4_moe_backend must be one of {_VALID_NVFP4_MOE_BACKENDS}, "
            f"got {raw!r}")
    return backend


def _parse_quant(model_dir: str, config: dict) -> QuantConfig:
    """Determine quantisation config from hf_quant_config.json or config.json."""

    # ---- Sidecar hf_quant_config.json ---------------------------------------
    hf_path = os.path.join(model_dir, "hf_quant_config.json")
    if os.path.exists(hf_path):
        with open(hf_path) as f:
            hq = json.load(f)
        q = hq.get("quantization", {})
        nvfp4_moe_backend = _parse_nvfp4_moe_backend(q)
        algo = (q.get("quant_algo") or "").upper()
        if "AWQ" in algo and "W4A16" in algo:
            # Drop exclusions that the checkpoint contradicts (false positives),
            # then augment with submodules ModelOpt actually left unquantized
            # (false negatives — typically visual tower / audio encoder).
            excluded = _effective_excluded_modules(
                model_dir, list(q.get("exclude_modules", [])))
            excluded.extend(
                m for m in _detect_modelopt_unquantized_linears(model_dir)
                if m not in excluded)
            return QuantConfig(
                quant_type=QUANT_INT4_AWQ_MODELOPT,
                group_size=int(q.get("group_size", 128)),
                excluded=excluded,
                nvfp4_moe_backend=nvfp4_moe_backend,
            )
        if algo == "MIXED_PRECISION":
            quantized_layers = q.get("quantized_layers", {})
            dominant, group_size, layer_overrides = _parse_mixed_precision(
                quantized_layers)
            return QuantConfig(
                quant_type=dominant,
                group_size=group_size,
                kv_cache_quant=_kv_norm(q.get("kv_cache_quant_algo", "")),
                excluded=_effective_excluded_modules(
                    model_dir, list(q.get("exclude_modules", []))),
                layer_overrides=layer_overrides,
                is_mixed_precision=True,
                nvfp4_moe_backend=nvfp4_moe_backend,
            )
        qt = _algo_to_quant_type(algo)
        gs = int(q.get("group_size", 1))
        if qt == QUANT_MXFP8 and gs == 1:
            gs = 32  # MXFP8 default block_size
        excluded = _effective_excluded_modules(
            model_dir, list(q.get("exclude_modules", [])))
        excluded.extend(
            m for m in _detect_modelopt_unquantized_linears(model_dir)
            if m not in excluded)
        return QuantConfig(
            quant_type=qt,
            group_size=gs,
            kv_cache_quant=_kv_norm(q.get("kv_cache_quant_algo", "")),
            excluded=excluded,
            nvfp4_moe_backend=nvfp4_moe_backend,
        )

    # ---- Embedded quantization_config in config.json ------------------------
    qc = config.get("quantization_config")
    if qc is None:
        return QuantConfig()
    nvfp4_moe_backend = _parse_nvfp4_moe_backend(qc)

    # Embedded block with ``quant_algo`` (export tool formats)
    if "quant_algo" in qc:
        algo = (qc.get("quant_algo") or "").upper()
        if "W4A16" in algo and "AWQ" in algo:
            return QuantConfig(
                quant_type=QUANT_INT4_AWQ_MODELOPT,
                group_size=int(qc.get("group_size", 128)),
                excluded=_effective_excluded_modules(
                    model_dir, list(qc.get("ignore", []))),
                nvfp4_moe_backend=nvfp4_moe_backend,
            )
        group_size = 1
        cg = qc.get("config_groups", {})
        if cg:
            first_group = next(iter(cg.values()), {})
            group_size = int(
                first_group.get("weights", {}).get("group_size", 1))
        kv = qc.get("kv_cache_scheme")
        kv_str = "fp8" if kv else None
        return QuantConfig(
            quant_type=_algo_to_quant_type(algo),
            group_size=group_size,
            kv_cache_quant=kv_str,
            excluded=_effective_excluded_modules(model_dir,
                                                 list(qc.get("ignore", []))),
            nvfp4_moe_backend=nvfp4_moe_backend,
        )

    # quant_method == awq (column-packed int4 checkpoints)
    if qc.get("quant_method") == "awq":
        return QuantConfig(
            quant_type=QUANT_INT4_AWQ,
            group_size=int(qc.get("group_size", 128)),
            excluded=_detect_unquantized_modules(model_dir),
            nvfp4_moe_backend=nvfp4_moe_backend,
        )

    # quant_method == gptq
    if qc.get("quant_method") == "gptq":
        return QuantConfig(
            quant_type=QUANT_INT4_GPTQ,
            group_size=int(qc.get("group_size", 128)),
            gptq_zero_point_offset=_detect_gptq_zero_point_offset(
                model_dir, qc),
            excluded=_detect_unquantized_modules(model_dir),
            nvfp4_moe_backend=nvfp4_moe_backend,
        )

    return QuantConfig(nvfp4_moe_backend=nvfp4_moe_backend)


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
    # MXFP8 per-block must be checked before generic FP8
    if "FP8_PB" in algo or "MXFP8" in algo:
        return QUANT_MXFP8
    if "FP8" in algo:
        return QUANT_FP8
    if "FP4" in algo or "NVFP4" in algo:
        return QUANT_NVFP4
    # W4A16_AWQ from ModelOpt unified checkpoints uses prepacked uint8 weights;
    # plain AWQ / INT4_AWQ from HuggingFace uses column-packed int32 qweight.
    if "W4A16" in algo and "AWQ" in algo:
        return QUANT_INT4_AWQ_MODELOPT
    if "AWQ" in algo or "INT4_AWQ" in algo:
        return QUANT_INT4_AWQ
    if "W8A8" in algo or "INT8" in algo:
        return QUANT_INT8_SQ
    return QUANT_FP16


def _parse_mixed_precision(quantized_layers: dict) -> "tuple[str, int, dict]":
    """Parse MIXED_PRECISION quantized_layers dict.

    Returns ``(dominant_quant_type, dominant_group_size, layer_overrides)``.
    ``layer_overrides`` maps **every** quantized module name to its quant-type
    string.  Modules not listed in ``quantized_layers`` are unquantized (FP16);
    ``make_linear`` falls back to FP16 when a module_name is absent from
    ``layer_overrides``.
    """
    from collections import Counter
    algo_count: Counter = Counter()
    algo_group_size: dict = {}
    for layer_cfg in quantized_layers.values():
        algo = layer_cfg.get("quant_algo", "").upper()
        algo_count[algo] += 1

        if algo not in algo_group_size:
            algo_group_size[algo] = int(layer_cfg.get("group_size", 1))
    if not algo_count:
        return QUANT_FP16, 1, {}
    dominant_algo = algo_count.most_common(1)[0][0]
    dominant_type = _algo_to_quant_type(dominant_algo)
    dominant_group_size = algo_group_size.get(dominant_algo, 1)
    # Expand fused projection keys (``self_attn.qkv_proj``,
    # ``mlp.gate_up_proj``) into the split names ``make_linear`` looks up
    # (``q_proj``/``k_proj``/``v_proj`` and ``gate_proj``/``up_proj``).
    layer_overrides: dict = {}
    for name, layer_cfg in quantized_layers.items():
        algo = layer_cfg.get("quant_algo", "").upper()
        short_name = _normalize_module_name(name)
        quant_type = _algo_to_quant_type(algo)
        if short_name.endswith(".self_attn.qkv_proj"):
            prefix = short_name[:-len("qkv_proj")]
            for proj in ("q_proj", "k_proj", "v_proj"):
                layer_overrides[f"{prefix}{proj}"] = quant_type
        elif short_name.endswith(".mlp.gate_up_proj"):
            prefix = short_name[:-len("gate_up_proj")]
            for proj in ("gate_proj", "up_proj"):
                layer_overrides[f"{prefix}{proj}"] = quant_type
        else:
            layer_overrides[short_name] = quant_type
    return dominant_type, dominant_group_size, layer_overrides


def _kv_norm(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return s.strip().lower() or None
