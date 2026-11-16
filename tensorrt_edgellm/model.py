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
Auto-dispatch model factory and parameter utilities.

``AutoModel.from_pretrained`` reads a checkpoint config, picks the right model
class, constructs it, and loads weights — the primary entry point for callers.

Custom model classes can be registered via :func:`register_model` to override
the default :class:`~models.default.modeling_default.CausalLM` for a given
``model_type`` string.
"""

import dataclasses
import itertools
import logging
import os
from typing import Callable, Dict, Set, Tuple, Type

import torch
import torch.nn as nn

from .checkpoint.checkpoint_utils import load_checkpoint_config_dicts
from .checkpoint.loader import iter_checkpoint_keys, load_weights
from .config import (_EAGLE_CANDIDATE_HEAD_KEY_PREFIXES,
                     _EAGLE_SKIPPED_CHECKPOINT_KEYS, QUANT_FP16,
                     QUANT_INT4_GPTQ, QUANT_INT8_GPTQ, ModelConfig,
                     _eagle_draft_feature_keys,
                     _is_gemma4_assistant_model_type, _is_gemma4_model_type,
                     _parse_eagle_target_layer_ids, make_dflash2_draft_config,
                     make_dflash_draft_config, make_domino_draft_config,
                     make_dspark_draft_config, make_jetspec_draft_config,
                     make_lilicorr_draft_config, make_mtp_draft_config,
                     module_quant_group_size, module_quant_type)
from .dflash import (DFlashVersion, dense_last_layer_modules,
                     resolve_dflash_contract)

__all__ = [
    "AutoModel", "load_model_config", "register_attention_scale_default",
    "register_model", "standard_attention_scale", "dtype_summary",
    "param_count"
]

AttentionScaleDefault = Callable[[int], float]
_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}
_ATTENTION_SCALE_DEFAULT_REGISTRY: Dict[str, AttentionScaleDefault] = {}


def _instantiate_model(model_class: Type[nn.Module],
                       config: ModelConfig,
                       device: str,
                       low_cpu_mem_usage: bool = False) -> nn.Module:
    if low_cpu_mem_usage:
        if device != "cpu":
            raise ValueError("low_cpu_mem_usage requires device='cpu'")
        with torch.device("meta"):
            return model_class(config)

    model = model_class(config)
    model.to(device)
    return model


def _materialize_checkpoint_defaults(model: nn.Module, device: str) -> None:
    materialize = getattr(model, "materialize_checkpoint_defaults", None)
    if callable(materialize):
        materialize(device)


# Quantization-scale buffers that model classes register with a safe ``ones``
# default but that not every checkpoint variant carries: official
# ``nvidia/*-FP8`` / ``*-NVFP4`` checkpoints omit the attention ``q_scale``
# (ModelOpt keeps the legacy ``qScale=1.0`` convention), and INT4-AWQ exports
# omit ``pre_quant_scale`` on the resmoothed q/k/v/gate/up projections. The
# normal (non-meta) load path simply keeps the registered ``ones`` default when
# the tensor is absent; under ``low_cpu_mem_usage`` that default lands on the
# meta device, so restore it explicitly before the strict materialization scan.
_OPTIONAL_SCALE_DEFAULT_BUFFERS = frozenset(
    ("q_scale", "k_scale", "v_scale", "pre_quant_scale"))


def _materialize_optional_scale_buffers(model: nn.Module, device: str) -> None:
    """Fill in optional ``ones``-default scale buffers left on the meta device.

    Mirrors the non-``low_cpu_mem_usage`` load path, where a checkpoint that
    omits one of these buffers keeps the module's registered ``ones`` default.
    Restricted to the known-optional scale buffer names so a genuinely missing
    required tensor (e.g. a weight) still trips the strict meta-tensor scan.
    """
    for module in model.modules():
        for name, buf in module.named_buffers(recurse=False):
            if (name in _OPTIONAL_SCALE_DEFAULT_BUFFERS and buf is not None
                    and buf.device.type == "meta"):
                module.register_buffer(
                    name, torch.ones(buf.shape, dtype=buf.dtype,
                                     device=device))


def standard_attention_scale(head_dim: int) -> float:
    return 1.0 / (float(head_dim)**0.5)


_MTP_BASE_MODEL_TYPES = frozenset({
    "qwen3",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen3_omni_next_text_moe",
    "qwen4_exp_text",
})
_MTP_DRAFT_MODEL_TYPES = frozenset({
    "qwen3",
    "qwen3_5_text",
    "qwen3_5_moe_text",
    "qwen3_omni_next_text_moe",
    "qwen4_exp_text",
})


def _is_mtp_base_supported(model_type: str) -> bool:
    return model_type in _MTP_BASE_MODEL_TYPES


def _is_mtp_draft_supported(model_type: str) -> bool:
    return model_type in _MTP_DRAFT_MODEL_TYPES


def _clamp_target_layer_ids_unique(layers: list,
                                   num_decoder_layers: int) -> list:
    """Clamp target_layer_ids into [0, num_decoder_layers), keeping them unique.

    Used for EAGLE3, whose runtime cross-check (validateEagleConfig) rejects
    duplicate conditioning layers -- unlike DFlash/MTP, a plain min() clamp
    isn't enough once truncation collapses two indices onto the same layer.
    Collisions are nudged to the nearest free slot instead.
    """
    used = set()
    result = []
    for index in layers:
        candidate = max(0, min(index, num_decoder_layers - 1))
        while candidate in used and candidate < num_decoder_layers - 1:
            candidate += 1
        while candidate in used and candidate > 0:
            candidate -= 1
        used.add(candidate)
        result.append(candidate)
    return result


def _inherit_dspark_base_metadata(config: ModelConfig,
                                  dspark_draft_dir: "str | None") -> None:
    if not dspark_draft_dir:
        return

    draft_cfg_path = os.path.join(dspark_draft_dir, "config.json")
    if not os.path.isfile(draft_cfg_path):
        raise FileNotFoundError(
            f"DSpark draft config does not exist: {draft_cfg_path}")
    _, draft_cfg = load_checkpoint_config_dicts(dspark_draft_dir)
    dspark_cfg = (draft_cfg.get("dspark_config")
                  or draft_cfg.get("dflash_config") or {})

    target_layer_ids = (dspark_cfg.get("target_layer_ids")
                        or draft_cfg.get("target_layer_ids")
                        or config.dspark_target_layer_ids)
    config.dspark_target_layer_ids = list(target_layer_ids)
    config.dspark_block_size = int(
        dspark_cfg.get("block_size",
                       draft_cfg.get("block_size", config.dspark_block_size)))
    default_mask_token_id = (4 if str(draft_cfg.get(
        "model_type", "")).startswith("gemma4") else
                             config.dspark_mask_token_id)
    config.dspark_mask_token_id = int(
        dspark_cfg.get("mask_token_id",
                       draft_cfg.get("mask_token_id", default_mask_token_id)))
    config.dspark_enable_confidence_head = bool(
        dspark_cfg.get(
            "enable_confidence_head",
            draft_cfg.get("enable_confidence_head",
                          config.dspark_enable_confidence_head)))
    config.dspark_confidence_head_with_markov = bool(
        dspark_cfg.get(
            "confidence_head_with_markov",
            draft_cfg.get("confidence_head_with_markov",
                          config.dspark_confidence_head_with_markov)))
    config.dspark_markov_head_type = str(
        dspark_cfg.get(
            "markov_head_type",
            draft_cfg.get("markov_head_type", config.dspark_markov_head_type)))
    config.dspark_markov_rank = int(
        dspark_cfg.get("markov_rank",
                       draft_cfg.get("markov_rank",
                                     config.dspark_markov_rank)))
    config.dspark_sample_from_anchor = bool(
        dspark_cfg.get(
            "sample_from_anchor",
            draft_cfg.get("sample_from_anchor",
                          config.dspark_sample_from_anchor)))


def _inherit_eagle_base_layer_ids(config: ModelConfig,
                                  eagle_draft_dir: str) -> None:
    """Set an EAGLE base's aux hidden-state layers from its paired draft."""
    draft_root, draft_cfg = load_checkpoint_config_dicts(eagle_draft_dir)
    if _eagle_draft_feature_keys(draft_cfg):
        # Drafts with fc_norm, norm_output or a candidate head name their aux
        # layers as HF hidden-state indices, which Gemma4 bases do not read.
        if _is_gemma4_model_type(config.model_type):
            raise ValueError(
                "A Gemma4 EAGLE base reads draft aux layers as decoder-layer "
                f"outputs; the draft at {eagle_draft_dir} uses fc_norm, "
                "norm_output or a candidate head, whose aux layer IDs are HF "
                "hidden-state indices.")
        layer_ids = _parse_eagle_target_layer_ids(
            draft_cfg, num_layers=config.num_hidden_layers)
        if not layer_ids and draft_cfg.get("eagle_config") is not None:
            raise ValueError(
                f"EAGLE draft at {eagle_draft_dir} declares eagle_config but "
                "no aux hidden-state layer IDs.")
        config.eagle3_target_layer_ids = layer_ids
        return
    config.eagle3_target_layer_ids = list(
        draft_cfg.get("target_layer_ids", []) or [])
    if (not config.eagle3_target_layer_ids
            and draft_root.get("speculators_model_type") == "eagle3"):
        config.eagle3_target_layer_ids = [
            2, config.num_hidden_layers // 2, config.num_hidden_layers - 3
        ]


def register_model(model_type: str, model_class: Type[nn.Module],
                   default_attention_scale: AttentionScaleDefault) -> None:
    """Register *model_class* as the handler for *model_type*.

    When :meth:`AutoModel.from_pretrained` encounters a checkpoint whose
    ``model_type`` field equals *model_type*, it instantiates *model_class*
    instead of the built-in :class:`~models.default.modeling_default.CausalLM`.

    Args:
        model_type:  Value of ``model_type`` in the checkpoint ``config.json``.
        model_class: ``nn.Module`` subclass; must accept a single
                     :class:`~config.ModelConfig` as its constructor argument.
        default_attention_scale: Function returning this family's default for
                     a given attention head dimension.
    """
    _MODEL_REGISTRY[model_type] = model_class
    _ATTENTION_SCALE_DEFAULT_REGISTRY[model_type] = default_attention_scale


def register_attention_scale_default(
        model_type: str,
        default_attention_scale: AttentionScaleDefault) -> None:
    """Register a default for a model type with special factory dispatch."""
    _ATTENTION_SCALE_DEFAULT_REGISTRY[model_type] = default_attention_scale


def load_model_config(model_dir: str) -> ModelConfig:
    """Load ModelConfig using the default declared by its model family."""
    default_attention_scale = _default_attention_scale_for_model_dir(model_dir)
    return ModelConfig.from_pretrained(model_dir, default_attention_scale)


def _default_attention_scale_for_model_dir(
        model_dir: str) -> AttentionScaleDefault:
    """Return the attention-scale default registered by checkpoint model_type."""
    root, llm_dict = load_checkpoint_config_dicts(model_dir)
    default_attention_scale = standard_attention_scale
    for config in (root, llm_dict):
        model_type = config.get("model_type")
        if model_type in _ATTENTION_SCALE_DEFAULT_REGISTRY:
            default_attention_scale = _ATTENTION_SCALE_DEFAULT_REGISTRY[
                model_type]
            break
    return default_attention_scale


class AutoModel:
    """HuggingFace-style factory that dispatches on ``model_type``."""

    @classmethod
    def from_pretrained(cls,
                        model_dir: str,
                        device: str = "cpu",
                        key_remap=None,
                        key_prefix: "str | None" = None,
                        eagle_base: bool = False,
                        eagle_draft_dir: "str | None" = None,
                        reduced_vocab_dir: "str | None" = None,
                        mtp_base: bool = False,
                        mtp_draft: bool = False,
                        tp_size: int = 1,
                        tp_rank: int = 0,
                        dflash_base: bool = False,
                        dflash_draft: bool = False,
                        dflash_draft_dir: "str | None" = None,
                        jetspec_base: bool = False,
                        jetspec_draft: bool = False,
                        jetspec_draft_dir: "str | None" = None,
                        dspark_base: bool = False,
                        dspark_draft: bool = False,
                        dspark_draft_dir: "str | None" = None,
                        gemma4_mtp_base: bool = False,
                        gemma4_mtp_draft: bool = False,
                        gemma4_kv_sharing_map: "list[dict] | None" = None,
                        gemma4_target_kv_cache_quant: "str | None" = None,
                        num_decoder_layers: "int | None" = None,
                        extra_configs: "dict | None" = None,
                        low_cpu_mem_usage: bool = False) -> nn.Module:
        """Construct and load a model from *model_dir*.

        Reads ``config.json`` via :class:`~config.ModelConfig`, looks up the
        model class in the registry (falling back to the built-in
        :class:`~models.default.modeling_default.CausalLM`), instantiates it,
        moves it to *device*, and loads safetensors weights.

        Args:
            model_dir:      Local HF checkpoint directory.
            device:         Target device (e.g. ``"cpu"``, ``"cuda:0"``).
            key_remap:      Optional callable ``(key: str) -> Optional[str]``.
                            Passed through to :func:`load_weights` for checkpoint
                            key remapping (e.g. TTS talker ``codec_embedding``
                            → ``embed_tokens``).
            key_prefix:     Explicit checkpoint key prefix to strip (e.g.
                            ``"talker."``).  Passed through to :func:`load_weights`.
            eagle_base:     When True, export as EAGLE3 base model with extra
                            tree-attention inputs and hidden_states output.
            eagle_draft_dir:
                            Optional EAGLE3 draft checkpoint directory. Gemma4
                            EAGLE3 uses it to select the draft-trained target
                            hidden layers for base hidden_states output.
            reduced_vocab_dir:
                            Optional directory containing ``vocab_map.safetensors``.
            mtp_base:       When True, export the standard Qwen3.5 text model as
                            the MTP base variant.
            mtp_draft:      When True, build the dedicated Qwen3.5 dense MTP
                            draft model from the base checkpoint config.
            tp_size:        Tensor-parallel world size.  When >1 the config
                            is reduced to per-rank shapes via
                            :meth:`ModelConfig.for_rank`, and weights
                            are sharded on assignment.  Default 1 = no TP.
            tp_rank:        This rank's index in [0, tp_size).
            dflash_base:    When True, export as DFlash base model.
            dflash_draft:   When True, build the DFlash draft model.
            dflash_draft_dir:
                            Path to the DFlash draft checkpoint directory.
            jetspec_base:   When True, export as JetSpec base model.
            jetspec_draft:  When True, build the JetSpec draft model.
            jetspec_draft_dir:
                            Path to the JetSpec draft checkpoint directory.
            dspark_base:    When True, export as DSpark base model.
            dspark_draft:   When True, build the DSpark draft backbone model.
            dspark_draft_dir:
                            Path to the DSpark draft checkpoint directory.
            gemma4_mtp_base:
                            Export a Gemma4 target checkpoint as the base
                            engine for paired Gemma4 MTP.
            gemma4_mtp_draft:
                            Export/load a paired Gemma4 assistant checkpoint.
            gemma4_kv_sharing_map:
                            Validated assistant-layer to target-layer map for
                            Gemma4 MTP draft runtime config.
            gemma4_target_kv_cache_quant:
                            Target/base KV-cache quantization mode inherited by
                            Gemma4 MTP draft inputs that alias target KV cache.
            num_decoder_layers:
                            When set, truncate the model to only the first N
                            decoder layers (few-layer numeric validation).
                            Supported for the plain default ``CausalLM`` path
                            (e.g. Qwen3), registered hybrid base models, MTP base,
                            and eagle/dflash base (their target_layer_ids are
                            clamped into the truncated range); rejected for
                            jetspec/dspark/gemma4-mtp and any draft variant. The
                            checkpoint's extra-layer weights are simply skipped by
                            the loader.
            low_cpu_mem_usage:
                            Initialize model tensors on the meta device and
                            materialize them from the checkpoint during load.

        Returns:
            Loaded ``nn.Module`` in eval mode.
        """
        from .models.default.modeling_default import CausalLM

        config = load_model_config(model_dir)
        if extra_configs:
            for key, value in extra_configs.items():
                setattr(config, key, value)
        # Qwen3-Omni Next ships both dense and sparse-MoE thinkers under the
        # same ``qwen3_omni_next_text`` model_type (the HF config is not
        # rewritten for the MoE variant). Detect MoE by ``num_experts > 0`` and
        # re-dispatch to the registered ``_moe`` class variant when one exists.
        if (getattr(config, "num_experts", 0)
                and f"{config.model_type}_moe" in _MODEL_REGISTRY):
            config.model_type = f"{config.model_type}_moe"
        if eagle_base:
            config.eagle_base = True
            if eagle_draft_dir:
                _inherit_eagle_base_layer_ids(config, eagle_draft_dir)
        if mtp_base or config.mtp_base:
            config.mtp_base = True
        if gemma4_mtp_base:
            config.gemma4_mtp_base = True
        if dflash_base:
            config.dflash_base = True
        dflash_version = DFlashVersion.V1
        dflash_projector = ""
        if config.dflash_base:
            if dflash_draft_dir:
                draft_root, draft_llm = load_checkpoint_config_dicts(
                    dflash_draft_dir)
                contract = resolve_dflash_contract(draft_root, draft_llm)
                dflash_version = contract.version
                config.dflash_version = contract.version
                config.dflash_target_layer_ids = list(
                    contract.target_layer_ids)
                config.dflash_block_size = contract.block_size
                config.dflash_mask_token_id = contract.mask_token_id
                if contract.is_domino:
                    dflash_projector = contract.projector_type
                    config.dflash_projector_type = contract.projector_type
                    config.domino_gru_hidden_dim = contract.gru_hidden_dim
                    config.domino_proj_dim = contract.proj_dim
                    config.domino_prefix_len = contract.prefix_len
                    config.domino_pool_size = min(contract.pool_size,
                                                  config.vocab_size)
                    config.domino_shift_label = contract.shift_label
                if contract.version == DFlashVersion.V2:
                    make_v2_draft_config = (make_lilicorr_draft_config
                                            if contract.is_lilicorr else
                                            make_dflash2_draft_config)
                    draft_config = make_v2_draft_config(
                        dflash_draft_dir,
                        _default_attention_scale_for_model_dir(
                            dflash_draft_dir))
                    if (config.hidden_size != draft_config.hidden_size
                            or config.vocab_size != draft_config.vocab_size):
                        raise ValueError(
                            "DFlash V2 base/draft hidden and vocabulary sizes must match."
                        )
                    invalid = [
                        index for index in contract.target_layer_ids
                        if index < 0 or index >= config.num_hidden_layers
                    ]
                    if invalid:
                        raise ValueError(
                            f"DFlash V2 target-layer IDs outside base model: {invalid}"
                        )
                    config.dflash2_target_layer_ids = list(
                        contract.target_layer_ids)
                    config.dflash2_block_size = contract.block_size
                    config.dflash2_mask_token_id = contract.mask_token_id
                    config.dflash2_is_causal = contract.is_causal
                    config.dflash2_conv_kernel_size = contract.conv_kernel_size
                    config.dflash2_conv_group_size = contract.conv_group_size
                    if contract.is_lilicorr:
                        _require_unscaled_lilicorr_embedding(config)
                        config.dflash_projector_type = contract.projector_type
                        for field in dataclasses.fields(draft_config):
                            if field.name.startswith("lilicorr_"):
                                setattr(config, field.name,
                                        getattr(draft_config, field.name))
                    else:
                        config.dflash2_selector_rank = contract.selector_rank
                        config.dflash2_selector_top_k = contract.selector_top_k
            if not config.dflash_target_layer_ids:
                config.dflash_target_layer_ids = [1, 8, 15, 22, 29]
        elif dflash_draft and dflash_draft_dir:
            draft_root, draft_llm = load_checkpoint_config_dicts(
                dflash_draft_dir)
            contract = resolve_dflash_contract(draft_root, draft_llm)
            dflash_version = contract.version
            dflash_projector = contract.projector_type
        if jetspec_base:
            config.jetspec_base = True
        if config.jetspec_base:
            if not config.jetspec_target_layer_ids and jetspec_draft_dir:
                import json
                draft_cfg_path = os.path.join(jetspec_draft_dir, "config.json")
                if os.path.isfile(draft_cfg_path):
                    with open(draft_cfg_path) as f:
                        draft_cfg = json.load(f)
                    jetspec_cfg = (draft_cfg.get("jetspec_config")
                                   or draft_cfg.get("dflash_config") or {})
                    config.jetspec_target_layer_ids = list(
                        jetspec_cfg.get("target_layer_ids",
                                        draft_cfg.get("target_layer_ids", []))
                        or [])
                    config.jetspec_block_size = int(
                        jetspec_cfg.get("block_size",
                                        draft_cfg.get("block_size", 16)))
                    config.jetspec_mask_token_id = int(
                        jetspec_cfg.get("mask_token_id",
                                        draft_cfg.get("mask_token_id",
                                                      151669)))
                    config.jetspec_causal_head = bool(
                        jetspec_cfg.get("causal_head",
                                        draft_cfg.get("causal_head", True)))
            if not config.jetspec_target_layer_ids:
                raise ValueError(
                    "jetspec_base requires JetSpec target_layer_ids; pass "
                    "jetspec_draft_dir or set jetspec_target_layer_ids.")
            if not config.jetspec_causal_head:
                raise ValueError(
                    "jetspec_base requires jetspec_config.causal_head=true.")
            config.dflash_target_layer_ids = list(
                config.jetspec_target_layer_ids)
            config.dflash_block_size = config.jetspec_block_size
            config.dflash_mask_token_id = config.jetspec_mask_token_id
        if dspark_base:
            config.dspark_base = True
        if config.dspark_base:
            _inherit_dspark_base_metadata(config, dspark_draft_dir)
            if not config.dspark_target_layer_ids:
                raise ValueError(
                    "dspark_base requires DSpark target_layer_ids; pass "
                    "dspark_draft_dir or set dspark_target_layer_ids.")
        if tp_size > 1:
            config = config.for_rank(tp_rank, tp_size)

        variant = _resolve_model_variant(config,
                                         eagle_base=eagle_base,
                                         mtp_base=config.mtp_base,
                                         mtp_draft=mtp_draft,
                                         dflash_base=config.dflash_base,
                                         dflash_draft=dflash_draft,
                                         dflash_version=dflash_version,
                                         dflash_projector=dflash_projector,
                                         jetspec_base=config.jetspec_base,
                                         jetspec_draft=jetspec_draft,
                                         dspark_base=config.dspark_base,
                                         dspark_draft=dspark_draft,
                                         gemma4_mtp_base=gemma4_mtp_base,
                                         gemma4_mtp_draft=gemma4_mtp_draft)

        # EAGLE3 draft: auto-detect from draft_vocab_size
        if variant == "eagle3_draft":
            from .models.eagle3.modeling_eagle3_draft import Eagle3DraftModel
            model_class = Eagle3DraftModel
            # Set up key remapping: midlayer -> layers.0, skip t2d
            if key_remap is None:
                key_remap = _eagle3_key_remap
        elif variant == "eagle4_draft":
            from .models.eagle4 import Eagle4DraftModel
            model_class = Eagle4DraftModel
            if key_remap is None and config.draft_candidate_count > 0:
                key_remap = _eagle4_key_remap
            elif key_remap is None:
                key_remap = _eagle4_dense_head_key_remap
        elif variant == "mtp_draft":
            is_nemotron_h_mtp = config.is_nemotron_h
            # TODO: support other model types
            if not (is_nemotron_h_mtp
                    or _is_mtp_draft_supported(config.model_type)):
                raise NotImplementedError(
                    "MTP draft is only supported for qwen3_5_text / "
                    "qwen3_5_moe_text / qwen3_omni_next_text_moe / qwen3 / "
                    "qwen4_exp_text / Nemotron-H checkpoints; got "
                    f"{config.model_type!r}.")
            draft_model_type = config.model_type
            tie_word_embeddings = config.tie_word_embeddings
            config = make_mtp_draft_config(config)
            if is_nemotron_h_mtp:
                from .models.nemotron_h import NemotronHMtpDraftModel
                model_class = NemotronHMtpDraftModel
            elif draft_model_type == "qwen3_omni_next_text_moe":
                from .models.qwen3_omni_next import \
                    Qwen3OmniNextMoeMtpDraftModel
                model_class = Qwen3OmniNextMoeMtpDraftModel
            elif draft_model_type == "qwen3_5_moe_text":
                from .models.qwen3_5_moe import Qwen3_5MoeMtpDraftModel
                model_class = Qwen3_5MoeMtpDraftModel
            elif draft_model_type == "qwen3":
                from .models.mtp import Qwen3MtpDraftModel
                model_class = Qwen3MtpDraftModel
            elif draft_model_type == "qwen4_exp_text":
                from .models.qwen4_exp import Qwen4ExpMtpDraftModel
                model_class = Qwen4ExpMtpDraftModel
            else:
                from .models.qwen3_5 import Qwen3_5MtpDraftModel
                model_class = Qwen3_5MtpDraftModel
            if key_remap is None:
                # Only borrow the base model's lm_head when the checkpoint
                # ships no dedicated ``mtp.lm_head.*`` tensors.
                share_base_lm_head = not any(
                    key.startswith("mtp.lm_head.")
                    for key in iter_checkpoint_keys(model_dir))

                def key_remap(key):
                    return _mtp_key_remap(
                        key,
                        tie_word_embeddings=tie_word_embeddings,
                        share_base_lm_head=share_base_lm_head)

        elif variant == "dflash_draft":
            if dflash_draft_dir is None:
                raise ValueError(
                    "dflash_draft requires dflash_draft_dir to be set.")
            from .models.dflash.modeling_dflash_draft import DFlashDraftModel
            base_config = config
            base_model_dir = model_dir
            base_tie_word_embeddings = base_config.tie_word_embeddings
            draft_has_lm_head = _checkpoint_has_dflash_lm_head(
                dflash_draft_dir)
            config = make_dflash_draft_config(
                dflash_draft_dir,
                _default_attention_scale_for_model_dir(dflash_draft_dir),
                target_vocab_size=base_config.vocab_size,
                quantize_w4a16_activations=True)
            config.dflash_overflow_fp32 = (
                base_config.model_type.startswith("gemma4")
                and config.model_type == "qwen3")
            if config.dflash_overflow_fp32:
                config.quant.excluded = list(
                    config.quant.excluded) + dense_last_layer_modules(
                        config.num_hidden_layers)
            if (base_config.model_type == "nemotron_h"
                    or any(key == "fc.weight_scale"
                           for key in iter_checkpoint_keys(dflash_draft_dir))):
                # A quantized fc runs at the checkpoint's native precision, like
                # a Nemotron-3.5 target (hidden states far inside FP16), rather
                # than the dense-FP16 + FP32 projection that guards Qwen3-8B
                # (target-hidden ~abs 2e4).
                config.dflash_fc_native_precision = True
                config.quant.excluded = [
                    e for e in config.quant.excluded if e != "fc"
                ]
            if not draft_has_lm_head:
                config = _inherit_dflash_lm_head_quant(config, base_config)
            model_class = DFlashDraftModel
            model_dir = dflash_draft_dir
            if key_remap is None:
                key_remap = (_muse_glimmer_dflash_key_remap
                             if config.model_type == "muse_glimmer_assistant"
                             else _dflash_key_remap)
        elif variant == "dflash2_draft":
            if dflash_draft_dir is None:
                raise ValueError(
                    "DFlash V2 draft requires dflash_draft_dir to be set.")
            from .models.dflash2 import DFlash2DraftModel
            base_config = config
            base_model_dir = model_dir
            base_tie_word_embeddings = base_config.tie_word_embeddings
            draft_has_lm_head = _checkpoint_has_dflash_lm_head(
                dflash_draft_dir)
            config = make_dflash2_draft_config(
                dflash_draft_dir,
                _default_attention_scale_for_model_dir(dflash_draft_dir))
            if not draft_has_lm_head:
                config = _inherit_dflash_lm_head_quant(config, base_config)
            model_class = DFlash2DraftModel
            model_dir = dflash_draft_dir
            if key_remap is None:
                key_remap = _dflash_key_remap
        elif variant == "lilicorr_draft":
            if dflash_draft_dir is None:
                raise ValueError(
                    "LiLiCorr draft requires dflash_draft_dir to be set.")
            if reduced_vocab_dir is not None:
                raise ValueError(
                    "LiLiCorr drafts do not support reduced-vocabulary export: "
                    "the candidate ids index the full target vocabulary and "
                    "its runtime embedding table.")
            from .models.lilicorr import LiLiCorrDraftModel
            base_config = config
            _require_unscaled_lilicorr_embedding(base_config)
            base_model_dir = model_dir
            base_tie_word_embeddings = base_config.tie_word_embeddings
            draft_has_lm_head = _checkpoint_has_dflash_lm_head(
                dflash_draft_dir, _dflash_key_remap)
            config = make_lilicorr_draft_config(
                dflash_draft_dir,
                _default_attention_scale_for_model_dir(dflash_draft_dir))
            if _checkpoint_has_draft_vocab_map(dflash_draft_dir):
                raise ValueError(
                    "LiLiCorr candidates require the full target vocabulary: "
                    "the draft checkpoint ships a d2t draft-to-target map")
            if config.vocab_size != base_config.vocab_size:
                raise ValueError(
                    "LiLiCorr draft vocabulary must equal the base vocabulary: "
                    "its candidate ids index the base embedding table "
                    f"(draft {config.vocab_size}, base {base_config.vocab_size})"
                )
            # fc and the head linears stay packed NVFP4 (W4A16 GEMM).
            config.dflash_fc_native_precision = True
            if not draft_has_lm_head:
                config = _inherit_dflash_lm_head_quant(config, base_config)
            model_class = LiLiCorrDraftModel
            model_dir = dflash_draft_dir
            if key_remap is None:
                key_remap = _dflash_key_remap
        elif variant == "domino_draft":
            if dflash_draft_dir is None:
                raise ValueError(
                    "Domino draft requires dflash_draft_dir to be set.")
            from .models.domino import DominoDraftModel
            base_config = config
            base_model_dir = model_dir
            base_tie_word_embeddings = base_config.tie_word_embeddings
            draft_has_lm_head = _checkpoint_has_dflash_lm_head(
                dflash_draft_dir, _domino_key_remap)
            config = make_domino_draft_config(
                dflash_draft_dir,
                _default_attention_scale_for_model_dir(dflash_draft_dir))
            if not draft_has_lm_head:
                config = _inherit_dflash_lm_head_quant(config, base_config)
            model_class = DominoDraftModel
            model_dir = dflash_draft_dir
            if key_remap is None:
                key_remap = _domino_key_remap
        elif variant == "jetspec_draft":
            if jetspec_draft_dir is None:
                raise ValueError(
                    "jetspec_draft requires jetspec_draft_dir to be set.")
            from .models.dflash.modeling_dflash_draft import DFlashDraftModel
            base_config = config
            base_model_dir = model_dir
            base_tie_word_embeddings = base_config.tie_word_embeddings
            draft_has_lm_head = _checkpoint_has_dflash_lm_head(
                jetspec_draft_dir)
            config = make_jetspec_draft_config(
                jetspec_draft_dir,
                _default_attention_scale_for_model_dir(jetspec_draft_dir))
            if not draft_has_lm_head:
                config = _inherit_dflash_lm_head_quant(config, base_config)
            model_class = DFlashDraftModel
            model_dir = jetspec_draft_dir
            if key_remap is None:
                key_remap = _dflash_key_remap
        elif variant == "gemma4_mtp_draft":
            if not _is_gemma4_assistant_model_type(config.root_model_type):
                raise ValueError(
                    "Gemma4 MTP draft requires a Gemma4 assistant checkpoint.")
            from .models.gemma4 import Gemma4AssistantForCausalLM
            config.gemma4_mtp_draft = True
            config.shares_target_kv = True
            config.has_own_kv_cache = False
            config.constant_draft_positions = True
            config.returns_feedback_hidden = True
            config.assistant_hidden_size = config.hidden_size
            config.kv_sharing_map = list(gemma4_kv_sharing_map or [])
            if gemma4_target_kv_cache_quant is not None:
                config.quant.kv_cache_quant = gemma4_target_kv_cache_quant
            model_class = Gemma4AssistantForCausalLM
        elif variant == "dspark_draft":
            if dspark_draft_dir is None:
                raise ValueError(
                    "dspark_draft requires dspark_draft_dir to be set.")
            from .models.dspark.modeling_dspark_draft import DSparkDraftModel
            base_config = config
            base_model_dir = model_dir
            base_tie_word_embeddings = base_config.tie_word_embeddings
            draft_has_lm_head = _checkpoint_has_dflash_lm_head(
                dspark_draft_dir, _dspark_key_remap)
            config = make_dspark_draft_config(
                dspark_draft_dir,
                _default_attention_scale_for_model_dir(dspark_draft_dir))
            if base_config.model_type == "nemotron_h":
                config.dspark_fc_native_precision = True
            if not draft_has_lm_head:
                config = _inherit_dflash_lm_head_quant(config, base_config)
            model_class = DSparkDraftModel
            model_dir = dspark_draft_dir
            if key_remap is None:
                key_remap = _dspark_key_remap
        else:
            if (variant == "mtp_base" and not config.is_nemotron_h
                    and not _is_mtp_base_supported(config.model_type)):
                raise NotImplementedError(
                    "MTP base is only supported for Qwen3.5 (text/MoE), dense "
                    "Qwen3, Qwen4-Exp and Nemotron-H checkpoints; got "
                    f"{config.model_type!r}.")
            if variant == "gemma4_mtp_base":
                if config.model_type not in ("gemma4", "gemma4_text",
                                             "gemma4_unified",
                                             "gemma4_unified_text"):
                    raise ValueError(
                        "Gemma4 MTP base requires a Gemma4 target checkpoint.")
                from .models.gemma4 import Gemma4ForCausalLM
                config.gemma4_mtp_base = True
                model_class = Gemma4ForCausalLM
            else:
                # DFlash base is supported for both Qwen3.5 hybrid (qwen3_5_text) and
                # dense Qwen3 (default CausalLM). Dense models use the Transformer's
                # dflash_target_layer_ids parameter to collect target-layer hidden states.
                model_class = _MODEL_REGISTRY.get(config.model_type, CausalLM)
                if (key_remap is None
                        and config.model_type == "hunyuan_v1_dense"):
                    key_remap = _hunyuan_key_remap
                if (key_remap is None
                        and str(config.model_type).startswith("muse_glimmer")):
                    from .models.muse_glimmer import MUSE_GLIMMER_KEY_REMAP
                    key_remap = MUSE_GLIMMER_KEY_REMAP

        # 4-layer numeric validation: truncate to the first N decoder
        # layers.  The whole pipeline is config-driven (the Transformer builds
        # layers from ``config.num_hidden_layers`` / ``config.layer_types``,
        # ``onnx_export_spec`` derives the KV / recurrent / conv I/O counts from
        # them, the runtime config.json is written from ``model.config``, and the
        # loader silently skips checkpoint keys for the dropped layers), so a
        # single config override is sufficient and the modeling code needs no
        # change.  This covers the plain default ``CausalLM`` path (e.g. Qwen3)
        # AND registered hybrid base models (Qwen3.5 linear+full / Gated DeltaNet,
        # Nemotron-H Mamba), which also build per-layer from ``layer_types``.
        #
        # Single-checkpoint MTP (``--mtp``) is also covered: its draft head reads
        # only the base's *last* hidden state, which stays well defined after
        # truncation.  (The draft was trained against the full stack, so its
        # proposals are near-useless on a truncated base and acceptance collapses
        # to one token per round -- that costs speed, not correctness, and the
        # few-layer comparison is about the base model's committed state.)
        #
        # Eagle3/DFlash base: their target_layer_ids index the *full* stack, so they
        # are clamped back into range below. DFlash tolerates duplicates after
        # clamping; EAGLE3's runtime cross-check does not, so its clamp dedupes.
        #
        # jetspec/dspark/gemma4-mtp stay out of scope: each names its own target-layer
        # contract. The MTP/eagle/dflash *draft* is likewise excluded; only the base
        # is compared against the golden.
        if num_decoder_layers is not None:
            if (jetspec_base or config.jetspec_base or dspark_base
                    or config.dspark_base or mtp_draft or dflash_draft
                    or jetspec_draft or dspark_draft or config.is_jetspec_draft
                    or config.is_dspark_draft or gemma4_mtp_base
                    or config.gemma4_mtp_base or gemma4_mtp_draft
                    or config.gemma4_mtp_draft):
                raise NotImplementedError(
                    "num_decoder_layers cannot be combined with the "
                    "jetspec/dspark/gemma4-mtp speculative-decoding variants "
                    "(single-checkpoint MTP base, and eagle/dflash base with "
                    "clamped target_layer_ids, are supported).")
            if not 1 <= num_decoder_layers <= config.num_hidden_layers:
                raise ValueError(
                    f"num_decoder_layers={num_decoder_layers} out of range "
                    f"[1, {config.num_hidden_layers}].")
            config.num_hidden_layers = num_decoder_layers
            if eagle_base or config.eagle_base:
                if num_decoder_layers < 3:
                    raise ValueError(
                        "EAGLE3 requires at least 3 decoder layers to provide "
                        "three distinct conditioning layers; got "
                        f"num_decoder_layers={num_decoder_layers}")
                config.eagle3_target_layer_ids = _clamp_target_layer_ids_unique(
                    config.eagle3_target_layer_ids, num_decoder_layers)
            if dflash_base or config.dflash_base:
                # Duplicates after clamping only make the draft head reread the same
                # shallow hidden state: degraded proposals, not a broken pipeline.
                config.dflash_target_layer_ids = [
                    min(i, num_decoder_layers - 1)
                    for i in config.dflash_target_layer_ids
                ]
                config.dflash2_target_layer_ids = [
                    min(i, num_decoder_layers - 1)
                    for i in config.dflash2_target_layer_ids
                ]
            # Keep per-layer lists consistent with the truncated layer count.
            if config.layer_types:
                config.layer_types = config.layer_types[:num_decoder_layers]
            logging.getLogger(__name__).info(
                "num_decoder_layers: truncated to first %d decoder layers",
                num_decoder_layers)

        model = _instantiate_model(model_class,
                                   config,
                                   device,
                                   low_cpu_mem_usage=low_cpu_mem_usage)
        if variant == "lilicorr_draft":
            _fill_lilicorr_load_sentinels(model)

        pre_repack_hook = None
        apply_reduced_vocab_after_load = False
        if reduced_vocab_dir is not None:
            from .vocab_reduction.onnx_export import (
                apply_reduced_vocab, load_reduced_vocab_map,
                should_apply_reduced_vocab_before_repacking)
            vocab_map = load_reduced_vocab_map(reduced_vocab_dir,
                                               vocab_size=config.vocab_size,
                                               device=device)
            if should_apply_reduced_vocab_before_repacking(model):

                def _apply_pre_repack_reduced_vocab(loaded_model: nn.Module):
                    apply_reduced_vocab(loaded_model, vocab_map)
                    loaded_model._reduced_vocab_dir = reduced_vocab_dir

                pre_repack_hook = _apply_pre_repack_reduced_vocab
            else:
                apply_reduced_vocab_after_load = True

        if variant in ("dflash_draft", "dflash2_draft", "lilicorr_draft",
                       "domino_draft", "jetspec_draft",
                       "dspark_draft") and not draft_has_lm_head:
            next_pre_repack_hook = pre_repack_hook

            def _load_pre_repack_dflash_lm_head(loaded_model: nn.Module):
                _load_dflash_lm_head(
                    loaded_model,
                    base_model_dir,
                    device,
                    tie_word_embeddings=base_tie_word_embeddings)
                if next_pre_repack_hook is not None:
                    next_pre_repack_hook(loaded_model)

            pre_repack_hook = _load_pre_repack_dflash_lm_head

        if config.ple_enabled:
            checkpoint_key_remap = key_remap

            def _skip_runtime_ple(key):
                mapped = checkpoint_key_remap(
                    key) if checkpoint_key_remap else key
                if mapped in ("model.embed_tokens_per_layer.weight",
                              "embed_tokens_per_layer.weight"):
                    return None
                return mapped

            key_remap = _skip_runtime_ple
        if variant == "dflash_draft":
            next_hook = pre_repack_hook

            def _dequantize_packed_fp16_linears(loaded_model: nn.Module):
                _dequantize_packed_dense_linears(loaded_model, model_dir,
                                                 device)
                if next_hook is not None:
                    next_hook(loaded_model)

            pre_repack_hook = _dequantize_packed_fp16_linears

        if variant == "domino_draft":
            next_domino_hook = pre_repack_hook

            def _load_pre_repack_domino_pool(loaded_model: nn.Module):
                _load_domino_pool_weights(loaded_model, model_dir, device)
                if next_domino_hook is not None:
                    next_domino_hook(loaded_model)

            pre_repack_hook = _load_pre_repack_domino_pool

        if variant == "lilicorr_draft":
            next_lilicorr_hook = pre_repack_hook
            lilicorr_key_remap = key_remap

            def _finish_pre_repack_lilicorr(loaded_model: nn.Module):
                if next_lilicorr_hook is not None:
                    next_lilicorr_hook(loaded_model)
                _finish_lilicorr_load(loaded_model, model_dir,
                                      lilicorr_key_remap)

            pre_repack_hook = _finish_pre_repack_lilicorr

        load_weights(model,
                     model_dir,
                     device=device,
                     key_remap=key_remap,
                     key_prefix=key_prefix,
                     pre_repack_hook=pre_repack_hook,
                     mapping=config.mapping)
        if low_cpu_mem_usage:
            _materialize_checkpoint_defaults(model, device)
            _materialize_optional_scale_buffers(model, device)
            meta_tensors = [
                name for name, tensor in (*model.named_parameters(),
                                          *model.named_buffers())
                if tensor.device.type == "meta"
            ]
            if meta_tensors:
                raise RuntimeError(
                    "Checkpoint did not materialize model tensors: " +
                    ", ".join(meta_tensors[:10]))
        if variant == "eagle4_draft":
            _require_eagle4_draft_tensors(model,
                                          iter_checkpoint_keys(model_dir))
        refresh_router_bias = getattr(model, "refresh_fp32_router_bias", None)
        if callable(refresh_router_bias):
            refresh_router_bias()
        if variant in ("dflash_draft", "dflash2_draft", "lilicorr_draft",
                       "domino_draft", "jetspec_draft", "dspark_draft"):
            if draft_has_lm_head:
                logging.getLogger(__name__).info(
                    "%s lm_head source: draft checkpoint buffers", {
                        "jetspec_draft": "JetSpec",
                        "dspark_draft": "DSpark",
                        "domino_draft": "Domino",
                        "lilicorr_draft": "LiLiCorr",
                    }.get(variant, "DFlash"))
        if apply_reduced_vocab_after_load:
            from .vocab_reduction.onnx_export import \
                apply_reduced_vocab_from_dir
            apply_reduced_vocab_from_dir(model, reduced_vocab_dir)

        # Post-load optimisation: fuse GDN input projections for every family whose mixer is
        # the shared GdnMixer.
        if (config.model_type
                in ("qwen3_5_text", "qwen3_5_moe_text", "qwen4_exp_text")
                and not mtp_draft):
            from .models.qwen3_5 import fuse_gdn_input_projections
            fuse_gdn_input_projections(model)

        # Fuse attention Q/K/V projections into one packed-QKV GEMM where eligible.
        from .models.default.modeling_default import fuse_qkv_projections
        fuse_qkv_projections(model)

        return model


def param_count(model: nn.Module) -> int:
    """Return total parameter element count (trainable and frozen)."""
    return sum(p.numel() for p in model.parameters())


def dtype_summary(model: nn.Module) -> Dict[str, int]:
    """Map dtype name -> number of parameter elements."""
    out: Dict[str, int] = {}
    for p in model.parameters():
        name = str(p.dtype).replace("torch.", "")
        out[name] = out.get(name, 0) + p.numel()
    return dict(sorted(out.items(), key=lambda x: -x[1]))


def _inherit_dflash_lm_head_quant(draft_config: ModelConfig,
                                  base_config: ModelConfig) -> ModelConfig:
    """Make an old DFlash draft build a loadable shared lm_head.

    Older DFlash drafts may omit ``lm_head.*`` and reuse the base output head.
    When that head is quantized, mirror its layout only for the draft ``lm_head``
    so the fallback loader can copy the base sidecar tensors.
    """
    lm_head_quant = module_quant_type("lm_head", base_config)
    if lm_head_quant == QUANT_FP16:
        return draft_config

    draft_quant = draft_config.quant
    base_quant = base_config.quant
    lm_head_group_size = module_quant_group_size("lm_head", base_config)

    layer_overrides = dict(draft_quant.layer_overrides)
    layer_overrides["lm_head"] = lm_head_quant
    excluded = [name for name in draft_quant.excluded if name != "lm_head"]
    layer_group_sizes = dict(draft_quant.layer_group_sizes)
    layer_group_sizes["lm_head"] = lm_head_group_size
    quant_updates = {
        "excluded": excluded,
        "layer_overrides": layer_overrides,
        "layer_group_sizes": layer_group_sizes,
    }
    if lm_head_quant in (QUANT_INT4_GPTQ, QUANT_INT8_GPTQ):
        quant_updates[
            "gptq_zero_point_offset"] = base_quant.gptq_zero_point_offset
    quant = dataclasses.replace(draft_quant, **quant_updates)
    return dataclasses.replace(draft_config, quant=quant)


# ---------------------------------------------------------------------------
# EAGLE3 helpers
# ---------------------------------------------------------------------------


def _resolve_model_variant(config: ModelConfig,
                           *,
                           eagle_base: bool,
                           mtp_base: bool,
                           mtp_draft: bool,
                           dflash_base: bool = False,
                           dflash_draft: bool = False,
                           dflash_version: DFlashVersion = DFlashVersion.V1,
                           dflash_projector: str = "",
                           jetspec_base: bool = False,
                           jetspec_draft: bool = False,
                           dspark_base: bool = False,
                           dspark_draft: bool = False,
                           gemma4_mtp_base: bool = False,
                           gemma4_mtp_draft: bool = False) -> str:
    """Resolve the requested model variant while keeping EAGLE3 behavior intact."""
    if eagle_base and mtp_base:
        raise ValueError("eagle_base and mtp_base cannot both be enabled.")
    if eagle_base and mtp_draft:
        raise ValueError("eagle_base and mtp_draft cannot both be enabled.")
    if mtp_base and mtp_draft:
        raise ValueError("mtp_base and mtp_draft cannot both be enabled.")
    if dflash_base and dflash_draft:
        raise ValueError(
            "DFlash base and draft variants cannot both be enabled.")
    if jetspec_base and jetspec_draft:
        raise ValueError(
            "jetspec_base and jetspec_draft cannot both be enabled.")
    if dspark_base and dspark_draft:
        raise ValueError(
            "dspark_base and dspark_draft cannot both be enabled.")
    if dflash_base and (eagle_base or mtp_base or mtp_draft or jetspec_base
                        or jetspec_draft or dspark_base or dspark_draft
                        or gemma4_mtp_base or gemma4_mtp_draft):
        raise ValueError(
            "dflash_base cannot be combined with other spec variants.")
    if dflash_draft and (eagle_base or mtp_base or mtp_draft or jetspec_base
                         or jetspec_draft or dspark_base or dspark_draft
                         or gemma4_mtp_base or gemma4_mtp_draft):
        raise ValueError(
            "dflash_draft cannot be combined with other spec variants.")
    if jetspec_base and (eagle_base or mtp_base or mtp_draft or dflash_base
                         or dflash_draft or dspark_base or dspark_draft
                         or gemma4_mtp_base or gemma4_mtp_draft):
        raise ValueError(
            "jetspec_base cannot be combined with other spec variants.")
    if jetspec_draft and (eagle_base or mtp_base or mtp_draft or dflash_base
                          or dflash_draft or dspark_base or dspark_draft
                          or gemma4_mtp_base or gemma4_mtp_draft):
        raise ValueError(
            "jetspec_draft cannot be combined with other spec variants.")
    if dspark_base and (eagle_base or mtp_base or mtp_draft or dflash_base
                        or dflash_draft or jetspec_base or jetspec_draft
                        or gemma4_mtp_base or gemma4_mtp_draft):
        raise ValueError(
            "dspark_base cannot be combined with other spec variants.")
    if dspark_draft and (eagle_base or mtp_base or mtp_draft or dflash_base
                         or dflash_draft or jetspec_base or jetspec_draft
                         or gemma4_mtp_base or gemma4_mtp_draft):
        raise ValueError(
            "dspark_draft cannot be combined with other spec variants.")
    if gemma4_mtp_base and (eagle_base or mtp_base or mtp_draft or dflash_base
                            or dflash_draft or jetspec_base or jetspec_draft
                            or dspark_base or dspark_draft):
        raise ValueError(
            "gemma4_mtp_base cannot be combined with other speculative variants."
        )
    if gemma4_mtp_draft and (eagle_base or mtp_base or mtp_draft or dflash_base
                             or dflash_draft or jetspec_base or jetspec_draft
                             or dspark_base or dspark_draft):
        raise ValueError(
            "gemma4_mtp_draft cannot be combined with other speculative variants."
        )
    if gemma4_mtp_base and gemma4_mtp_draft:
        raise ValueError(
            "gemma4_mtp_base and gemma4_mtp_draft cannot both be enabled.")
    if config.is_eagle3_draft:
        if mtp_base or mtp_draft:
            raise ValueError(
                "EAGLE3 draft checkpoints cannot be loaded as Qwen3.5 MTP variants."
            )
        if config.is_eagle4_draft:
            return "eagle4_draft"
        return "eagle3_draft"
    if gemma4_mtp_draft:
        return "gemma4_mtp_draft"
    if dflash_draft and dflash_projector == "domino":
        return "domino_draft"
    if dflash_draft and dflash_projector == "lilicorr":
        return "lilicorr_draft"
    if dflash_draft and dflash_version == DFlashVersion.V2:
        return "dflash2_draft"
    if dflash_draft:
        return "dflash_draft"
    if dflash_base:
        return "dflash_base"
    if jetspec_draft:
        return "jetspec_draft"
    if jetspec_base:
        return "jetspec_base"
    if dspark_draft:
        return "dspark_draft"
    if dspark_base:
        return "dspark_base"
    if mtp_draft:
        return "mtp_draft"
    if mtp_base:
        return "mtp_base"
    if gemma4_mtp_base:
        return "gemma4_mtp_base"
    if eagle_base:
        return "eagle_base"
    return "llm"


def _hunyuan_key_remap(key: str) -> "str | None":
    """Remap HunYuan V1 checkpoint keys to the default CausalLM module tree.

    HunYuan names its per-head QK norms ``query_layernorm`` / ``key_layernorm``
    where the default :class:`Attention` module uses ``q_norm`` / ``k_norm``.
    """
    key = key.replace(".self_attn.query_layernorm.", ".self_attn.q_norm.")
    key = key.replace(".self_attn.key_layernorm.", ".self_attn.k_norm.")
    return key


def _eagle3_key_remap(key: str) -> "str | None":
    """Remap EAGLE3 draft checkpoint keys.

    Handles all known EAGLE3 draft checkpoint variations:
    - ``t2d`` keys are skipped (but ``d2t`` is kept).
    - ``target_model.*`` keys are skipped (multi-target training artifact).
    - ``midlayer.*`` -> ``layers.0.*``
    - ``qkv_proj.{q,k,v}_proj`` -> ``{q,k,v}_proj`` (flatten old pipeline
      ``EdgeLLMAttention`` wrapper nesting, used by quantized checkpoints).
    - ``._pre_quant_scale`` -> ``.pre_quant_scale`` (modelopt internal naming;
      normally stripped by ``postprocess_state_dict()`` but not by per-module
      export via ``_export_quantized_weight()``).
    """
    if "t2d" in key and "d2t" not in key:
        return None  # skip t2d
    if key.startswith("target_model."):
        return None  # skip multi-target training artifact
    key = key.replace("midlayer.", "layers.0.")
    key = key.replace("qkv_proj.q_proj", "q_proj")
    key = key.replace("qkv_proj.k_proj", "k_proj")
    key = key.replace("qkv_proj.v_proj", "v_proj")
    key = key.replace("._pre_quant_scale", ".pre_quant_scale")
    return key


def _eagle4_key_remap(key: str) -> "str | None":
    """:func:`_eagle3_key_remap` for drafts with a low-rank candidate head.

    ``embed_tokens.weight`` is skipped (drafts read the base embedding), and the
    candidate head factors load into ``candidate_head``:
    ``efficient_lm_head_down_proj.*`` -> ``candidate_head.down_proj.*`` and
    ``efficient_lm_head.*`` -> ``candidate_head.up_proj.*``.
    """
    if key in _EAGLE_SKIPPED_CHECKPOINT_KEYS:
        return None
    for prefix, module_prefix in _EAGLE_CANDIDATE_HEAD_KEY_PREFIXES:
        if key.startswith(prefix):
            return module_prefix + key[len(prefix):]
    return _eagle3_key_remap(key)


def _eagle4_dense_head_key_remap(key: str) -> "str | None":
    """:func:`_eagle4_key_remap` for drafts exported with the dense head.

    The low-rank candidate head factors are skipped.
    """
    module_key = _eagle4_key_remap(key)
    if module_key is not None and module_key.startswith("candidate_head."):
        return None
    return module_key


def _checkpoint_tensor_names(model: nn.Module) -> Tuple[Set[str], Set[str]]:
    """Return the model's tensor names as checkpoint names, plus unread names.

    NVFP4-A16 linears repack ``weight``, ``weight_scale`` and
    ``weight_scale_2`` into plugin buffers on load and keep no buffer for the
    W4A4 ``input_scale`` (the second set).
    """
    from .models.linear import NVFP4A16Linear

    repacked = {
        "qweight": "weight",
        "block_scales": "weight_scale",
        "global_scale": "weight_scale_2",
    }
    names = set(model.state_dict())
    unread = set()
    for module_name, module in model.named_modules():
        if not isinstance(module, NVFP4A16Linear):
            continue
        prefix = f"{module_name}." if module_name else ""
        for buffer_name, checkpoint_name in repacked.items():
            if prefix + buffer_name in names:
                names.discard(prefix + buffer_name)
                names.add(prefix + checkpoint_name)
        unread.add(prefix + "input_scale")
    return names, unread


def _require_eagle4_draft_tensors(model: nn.Module, checkpoint_keys) -> None:
    """Raise unless checkpoint and draft-model tensors match one to one.

    The loader only counts skipped keys, so a misnamed or missing tensor would
    otherwise leave a module at its initial value. A draft model without a
    candidate head skips the checkpoint's candidate head factors.
    """
    has_candidate_head = getattr(model, "candidate_head", None) is not None
    key_remap = (_eagle4_key_remap
                 if has_candidate_head else _eagle4_dense_head_key_remap)
    remapped = {}
    for key in checkpoint_keys:
        module_key = key_remap(key)
        if module_key is not None:
            remapped[module_key] = key
    state = model.state_dict()
    names, unread = _checkpoint_tensor_names(model)
    unused = sorted(key for module_key, key in remapped.items()
                    if module_key not in names and module_key not in unread)
    missing = names - remapped.keys()
    # A checkpoint may omit d2t only when the module default is already the
    # identity map, i.e. the draft vocabulary is the full vocabulary.
    d2t = state.get("d2t")
    if (d2t is not None and d2t.device.type != "meta" and torch.equal(
            d2t.reshape(-1),
            torch.arange(d2t.numel(), dtype=d2t.dtype, device=d2t.device))):
        missing.discard("d2t")
    missing = sorted(missing)
    if unused or missing:
        raise ValueError(
            "EAGLE draft checkpoint does not match the draft model. "
            f"Checkpoint tensors without a model tensor: {unused or 'none'}. "
            f"Model tensors without a checkpoint tensor: {missing or 'none'}.")


def _dequantize_packed_dense_linears(model: nn.Module, model_dir: str,
                                     device: str) -> None:
    """Dequantize dense ``FP16Linear`` modules whose checkpoint weight is packed NVFP4.

    The generic loader leaves the packed ``uint8 [out, in/2]`` payload in the
    module; rebuild the dense weight from the packed weight, block scales and
    global scale, as the checkpoint-direct builder does.
    """
    from .checkpoint.loader import _build_shard_map
    from .models.linear import FP16Linear
    from .models.ops import _nvfp4_dequantize_eager

    shard_map = _build_shard_map(model_dir)
    for name, module in model.named_modules():
        if (isinstance(module, FP16Linear)
                and module.weight.dtype == torch.uint8):
            scale, global_scale = (_read_dflash_checkpoint_tensor(
                shard_map[key], key, device)
                                   for key in (f"{name}.weight_scale",
                                               f"{name}.weight_scale_2"))
            module.weight = nn.Parameter(_nvfp4_dequantize_eager(
                module.weight, scale, global_scale, 16),
                                         requires_grad=False)


def _load_dflash_lm_head(model: nn.Module,
                         base_model_dir: str,
                         device: str,
                         *,
                         tie_word_embeddings: bool = True) -> None:
    """Load a DFlash-style draft lm_head from the base model checkpoint.

    JetSpec reuses the same DFlash draft-head module and official JetSpec
    draft checkpoints may also omit ``lm_head.*``.

    Deterministic source selection (matching MTP pattern):
      1. Explicit ``lm_head.weight`` from the base checkpoint.
      2. Embedding fallback *only* when ``tie_word_embeddings=True``.
      3. Otherwise fail loudly — untied models must not use embeddings.

    This helper is used only for DFlash-style draft checkpoints that do not carry
    ``lm_head.*`` tensors.  Quantized checkpoints, and dense checkpoints that
    explicitly save ``lm_head.weight``, are loaded by the generic checkpoint
    loader and must not be overwritten from the base embedding table.
    """
    import logging
    import os

    import torch

    from .models.linear import FP16Linear

    logger = logging.getLogger(__name__)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        logger.warning("DFlash-style draft model has no lm_head; skipping.")
        return

    from .checkpoint.loader import _build_shard_map
    shard_map = _build_shard_map(base_model_dir)

    if not isinstance(lm_head, FP16Linear):
        _load_dflash_quantized_lm_head(model, lm_head, shard_map,
                                       base_model_dir, device)
        return

    # --- Determine source key with strict priority ---
    lm_head_candidates = [
        "lm_head.weight",
        "model.lm_head.weight",
        "language_model.lm_head.weight",
        "model.language_model.lm_head.weight",
    ]
    embed_candidates = [
        "model.embed_tokens.weight",
        "embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
    ]

    # Priority 1: explicit lm_head.weight from base checkpoint
    source_key = None
    source_type = None
    for cand in lm_head_candidates:
        if cand in shard_map:
            source_key = cand
            source_type = "lm_head"
            break

    # Priority 2: embedding fallback only if tie_word_embeddings
    if source_key is None:
        if tie_word_embeddings:
            for cand in embed_candidates:
                if cand in shard_map:
                    source_key = cand
                    source_type = "tied_embedding"
                    break
        else:
            raise ValueError(
                "DFlash lm_head: base model at %s has "
                "tie_word_embeddings=False but no lm_head.weight found. "
                "Cannot safely fall back to embed_tokens." % base_model_dir)

    if source_key is None:
        raise ValueError(
            "Cannot find lm_head.weight or embed_tokens.weight in "
            "base model at %s." % base_model_dir)

    shard_path = shard_map[source_key]
    if source_type == "tied_embedding":
        logger.info("DFlash lm_head source: %s (tied fallback) from %s",
                    source_key, os.path.basename(shard_path))
    else:
        logger.info("DFlash lm_head source: %s from %s", source_key,
                    os.path.basename(shard_path))

    source_weight = _read_dflash_checkpoint_tensor(shard_path, source_key,
                                                   device)

    # --- Copy weight into model's dense lm_head ---
    if source_weight.shape != lm_head.weight.shape:
        raise ValueError(
            f"DFlash lm_head shape mismatch: source={source_weight.shape} "
            f"vs lm_head={lm_head.weight.shape}")
    with torch.no_grad():
        lm_head.weight.copy_(source_weight.to(lm_head.weight.dtype))


def _load_dflash_quantized_lm_head(model: nn.Module, lm_head: nn.Module,
                                   shard_map: Dict[str, str],
                                   base_model_dir: str, device: str) -> None:
    """Load quantized lm_head sidecar tensors shared from the base checkpoint."""
    import logging
    import os

    from .checkpoint.loader import _set_tensor

    logger = logging.getLogger(__name__)
    source_prefixes = (
        "lm_head",
        "model.lm_head",
        "language_model.lm_head",
        "model.language_model.lm_head",
    )
    target_state = lm_head.state_dict()
    missing = []
    source_by_target = {}
    optional_tensors = {"g_idx", "int4_act_perm", "pre_quant_scale"}

    for target_name in target_state:
        source_names = [target_name]
        if target_name == "weight_scale":
            source_names.append("weight_scale_inv")
        source_key = next((f"{prefix}.{source_name}"
                           for prefix in source_prefixes
                           for source_name in source_names
                           if f"{prefix}.{source_name}" in shard_map), None)
        if source_key is None:
            if target_name not in optional_tensors:
                missing.append(target_name)
            continue
        source_by_target[target_name] = source_key

    if missing:
        raise ValueError(
            "DFlash quantized lm_head fallback requires base checkpoint tensors "
            "for %s; missing %s in %s." % (", ".join(
                target_state.keys()), ", ".join(missing), base_model_dir))

    loaded_tensors = []
    for target_name, source_key in source_by_target.items():
        target_tensor = target_state[target_name]

        source_tensor = _read_dflash_checkpoint_tensor(shard_map[source_key],
                                                       source_key, device)
        source_tensor = _normalize_dflash_lm_head_tensor_shape(
            target_name, source_tensor, target_tensor)
        loaded_tensors.append((target_name, source_key, source_tensor))

    loaded = []
    for target_name, source_key, source_tensor in loaded_tensors:
        if not _set_tensor(model,
                           f"lm_head.{target_name}",
                           source_tensor,
                           mapping=model.config.mapping):
            raise ValueError(
                f"Failed to assign DFlash lm_head tensor {target_name!r}.")
        loaded.append(source_key)

    logger.info(
        "DFlash lm_head source: quantized base checkpoint buffers from %s",
        ", ".join(sorted({os.path.basename(shard_map[key])
                          for key in loaded})))


def _read_dflash_checkpoint_tensor(shard_path: str, key: str, device: str):
    import torch
    from safetensors import safe_open

    if shard_path.endswith(".bin"):
        state = torch.load(shard_path, map_location=device, weights_only=True)
        return state[key]
    with safe_open(shard_path, framework="pt", device=device) as f:
        return f.get_tensor(key)


def _normalize_dflash_lm_head_tensor_shape(target_name: str, source, target):
    if source.shape == target.shape:
        return source
    if source.numel() == 1 and target.numel() == 1:
        return source.reshape(target.shape)
    raise ValueError(
        "DFlash quantized lm_head shape mismatch for %s: source=%s vs "
        "lm_head=%s" % (target_name, source.shape, target.shape))


def _checkpoint_has_dflash_lm_head(model_dir: str, key_remap=None) -> bool:
    """Return whether a cached-draft checkpoint owns lm_head tensors.

    ``key_remap`` defaults to the DFlash remap; DSpark passes its own so the
    check runs against the same key view the loader will use.
    """
    from .checkpoint.loader import _build_shard_map

    remap = key_remap if key_remap is not None else _dflash_key_remap
    for key in _build_shard_map(model_dir):
        mapped = remap(key)
        if mapped is not None and mapped.startswith("lm_head."):
            return True
    return False


def _checkpoint_has_draft_vocab_map(model_dir: str) -> bool:
    """Return whether a draft checkpoint ships a ``d2t`` reduced-vocabulary map."""
    from .checkpoint.loader import _build_shard_map

    return "d2t" in _build_shard_map(model_dir)


def _dflash_key_remap(key: str) -> "str | None":
    """Remap DFlash draft checkpoint keys."""
    if "rotary_emb" in key:
        return None
    if key == "d2t":
        return "draft_vocab_map"
    if key == "t2d":
        return None
    return key


_LILICORR_SENTINEL_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _require_unscaled_lilicorr_embedding(base_config) -> None:
    """Reject a base whose runtime embedding table carries a folded scale.

    The LiLiCorr draft gathers raw rows of that table for its candidates.
    """
    from .checkpoint.checkpoint_utils import runtime_embedding_scale
    scale = runtime_embedding_scale(base_config)
    if scale != 1.0:
        raise ValueError(
            "LiLiCorr drafts gather raw rows of the runtime embedding table, "
            f"but this base pre-scales it by {scale}")


def _lilicorr_sentinel_tensors(model: nn.Module):
    """Yield floating tensors that the draft or base checkpoint must fill."""
    for name, tensor in itertools.chain(model.named_parameters(),
                                        model.named_buffers()):
        if (tensor.dtype in _LILICORR_SENTINEL_DTYPES and name.rsplit(
                ".", 1)[-1] not in _OPTIONAL_SCALE_DEFAULT_BUFFERS):
            yield name, tensor


def _fill_lilicorr_load_sentinels(model: nn.Module) -> None:
    """NaN-fill checkpoint-backed floating tensors so a missing tensor cannot pass as data."""
    with torch.no_grad():
        for _, tensor in _lilicorr_sentinel_tensors(model):
            if tensor.device.type != "meta":
                tensor.fill_(float("nan"))


def _assert_lilicorr_weights_loaded(model: nn.Module) -> None:
    """Raise ``ValueError`` naming every LiLiCorr draft tensor the load left unfilled.

    Unfilled means still holding the NaN load sentinel, still on the meta
    device, or an FP16 linear whose weight is still packed NVFP4.
    """
    from .models.linear import FP16Linear

    unfilled = [
        name for name, tensor in _lilicorr_sentinel_tensors(model)
        if tensor.device.type == "meta" or bool(torch.isnan(tensor).any())
    ]
    unfilled += [
        f"{name}.weight" for name, module in model.named_modules()
        if isinstance(module, FP16Linear)
        and not module.weight.is_floating_point()
    ]
    if unfilled:
        raise ValueError(
            "LiLiCorr draft tensors were not loaded from the checkpoint: " +
            ", ".join(unfilled))


def _check_lilicorr_coverage(model: nn.Module, shard_map: Dict[str, str],
                             key_remap) -> None:
    """Raise unless every draft tensor is consumed and every packed model tensor is provided.

    The generic loader skips unknown keys and leaves absent tensors at their
    initial values. NaN sentinels cover the floating tensors; this covers
    stray checkpoint keys and the packed NVFP4 weights and block scales.
    """
    targets = dict(
        itertools.chain(model.named_parameters(), model.named_buffers()))
    provided = set()
    unconsumed = []
    consumed = 0
    for key in shard_map:
        mapped = key_remap(key)
        if mapped is None:
            continue
        if mapped not in targets:
            unconsumed.append(key)
            continue
        provided.add(mapped)
        consumed += 1
    missing = [
        name for name, tensor in targets.items() if
        tensor.dtype not in _LILICORR_SENTINEL_DTYPES and name not in provided
        and not name.startswith("lm_head.") and name != "draft_vocab_map"
    ]
    if unconsumed or missing:
        raise ValueError(
            "LiLiCorr draft checkpoint does not match the model: unconsumed "
            f"checkpoint tensors [{', '.join(unconsumed)}]; missing packed "
            f"tensors [{', '.join(missing)}]")
    logging.getLogger(__name__).info(
        "LiLiCorr: consumed %d of %d draft checkpoint tensors", consumed,
        len(shard_map))


def _finish_lilicorr_load(model: nn.Module, draft_dir: str, key_remap) -> None:
    """Verify that every draft tensor loaded and build the lattice attention bias."""
    from .checkpoint.loader import _build_shard_map

    _check_lilicorr_coverage(model, _build_shard_map(draft_dir), key_remap)
    model.lilicorr.materialize()
    _assert_lilicorr_weights_loaded(model)


def _muse_glimmer_dflash_key_remap(key: str) -> "str | None":
    """Map the Muse-Glimmer assistant fusion projector onto DFlash names."""
    if key.startswith("encoder.fc."):
        return key.removeprefix("encoder.")
    if key.startswith("encoder.output_norm_enc."):
        return "hidden_norm." + key.removeprefix("encoder.output_norm_enc.")
    return _dflash_key_remap(key)


def _domino_key_remap(key: str) -> "str | None":
    """Remap Domino draft checkpoint keys for the backbone engine.

    The GRU tensors ship as a runtime sidecar and embed_proj is split by
    ``_load_domino_pool_weights``, so the generic loader skips them.
    """
    if "rotary_emb" in key:
        return None
    if key.startswith(("prefix_gru.", "embed_proj.")):
        return None
    return key


def _load_domino_pool_weights(model: nn.Module, draft_dir: str,
                              device: str) -> None:
    """Fill the in-engine Domino pool weights from the draft checkpoint.

    The checkpoint stores embed_proj.0 as one [proj_dim, H + gru_hidden]
    matrix; the engine consumes only the draft-hidden half (the state half
    ships in the runtime sidecar), and embed_proj.2 becomes the codebook.
    """
    import torch

    from .checkpoint.loader import _build_shard_map

    config = model.config
    shard_map = _build_shard_map(draft_dir)
    for key in ("embed_proj.0.weight", "embed_proj.2.weight"):
        if key not in shard_map:
            raise ValueError(
                f"Domino draft checkpoint at {draft_dir} is missing {key}.")

    w0 = _read_dflash_checkpoint_tensor(shard_map["embed_proj.0.weight"],
                                        "embed_proj.0.weight", device)
    expected_w0 = (config.domino_proj_dim,
                   config.hidden_size + config.domino_gru_hidden_dim)
    if tuple(w0.shape) != expected_w0:
        raise ValueError(
            f"Domino embed_proj.0.weight must have shape {expected_w0}, "
            f"got {tuple(w0.shape)}")
    codebook = _read_dflash_checkpoint_tensor(shard_map["embed_proj.2.weight"],
                                              "embed_proj.2.weight", device)
    expected_codebook = (config.vocab_size, config.domino_proj_dim)
    if tuple(codebook.shape) != expected_codebook:
        raise ValueError(
            f"Domino embed_proj.2.weight must have shape {expected_codebook}, "
            f"got {tuple(codebook.shape)}")

    pool = model.domino_pool
    with torch.no_grad():
        pool.z_projection.weight.copy_(w0[:, :config.hidden_size].to(
            pool.z_projection.weight.dtype))
        pool.codebook.copy_(codebook.to(pool.codebook.dtype))


def _dspark_key_remap(key: str) -> "str | None":
    """Remap DSpark draft checkpoint keys for the backbone engine.

    Markov and confidence tensors are exported as sidecars rather than baked
    into the backbone engine, so the generic loader may skip them.
    """
    if "rotary_emb" in key:
        return None
    if key == "d2t":
        return "draft_vocab_map"
    if key == "t2d":
        return None
    return key


def _mtp_key_remap(key: str,
                   *,
                   tie_word_embeddings: bool,
                   share_base_lm_head: bool = False) -> "str | None":
    """Remap MTP checkpoint keys for the draft model.

    ``share_base_lm_head`` sources the draft head from the base model
    (``lm_head.*`` tensors, or the embedding table when tied); enable it only
    when the checkpoint has no ``mtp.lm_head.*`` of its own.
    """
    if key.startswith("mtp."):
        return key[len("mtp."):]
    if share_base_lm_head and key.startswith("lm_head."):
        return key
    if share_base_lm_head and tie_word_embeddings and key in (
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight"):
        return "lm_head.weight"
    return None
