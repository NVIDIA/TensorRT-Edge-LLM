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
import logging
import os
from typing import Callable, Dict, Type

import torch.nn as nn

from .checkpoint.checkpoint_utils import load_checkpoint_config_dicts
from .checkpoint.loader import load_weights
from .config import (QUANT_FP16, QUANT_INT4_AWQ, QUANT_INT4_AWQ_MODELOPT,
                     QUANT_INT4_GPTQ, QUANT_MXFP8, QUANT_NVFP4, ModelConfig,
                     make_dflash_draft_config, make_mtp_draft_config,
                     module_quant_type)

__all__ = [
    "AutoModel", "load_model_config", "register_attention_scale_default",
    "register_model", "standard_attention_scale", "dtype_summary",
    "param_count"
]

AttentionScaleDefault = Callable[[int], float]
_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}
_ATTENTION_SCALE_DEFAULT_REGISTRY: Dict[str, AttentionScaleDefault] = {}


def standard_attention_scale(head_dim: int) -> float:
    return 1.0 / (float(head_dim)**0.5)


_QWEN3_5_MTP_BASE_MODEL_TYPES = frozenset({
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
})
_QWEN3_5_MTP_DRAFT_MODEL_TYPES = frozenset({
    "qwen3_5_text",
})


def _is_qwen3_5_mtp_base_supported(model_type: str) -> bool:
    return model_type in _QWEN3_5_MTP_BASE_MODEL_TYPES


def _is_qwen3_5_mtp_draft_supported(model_type: str) -> bool:
    return model_type in _QWEN3_5_MTP_DRAFT_MODEL_TYPES


_GROUP_SIZE_LM_HEAD_QUANTS = frozenset({
    QUANT_INT4_AWQ,
    QUANT_INT4_AWQ_MODELOPT,
    QUANT_INT4_GPTQ,
    QUANT_MXFP8,
    QUANT_NVFP4,
})


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
    root, llm_dict = load_checkpoint_config_dicts(model_dir)
    default_attention_scale = standard_attention_scale
    for config in (root, llm_dict):
        model_type = config.get("model_type")
        if model_type in _ATTENTION_SCALE_DEFAULT_REGISTRY:
            default_attention_scale = _ATTENTION_SCALE_DEFAULT_REGISTRY[
                model_type]
            break
    return ModelConfig.from_pretrained(model_dir, default_attention_scale)


class AutoModel:
    """HuggingFace-style factory that dispatches on ``model_type``."""

    @classmethod
    def from_pretrained(cls,
                        model_dir: str,
                        device: str = "cpu",
                        key_remap=None,
                        key_prefix: "str | None" = None,
                        eagle_base: bool = False,
                        reduced_vocab_dir: "str | None" = None,
                        mtp_base: bool = False,
                        mtp_tree_base: bool = False,
                        mtp_draft: bool = False,
                        tp_size: int = 1,
                        tp_rank: int = 0,
                        dflash_base: bool = False,
                        dflash_tree_base: bool = False,
                        dflash_draft: bool = False,
                        dflash_draft_dir: "str | None" = None,
                        gemma4_mtp_base: bool = False,
                        gemma4_mtp_draft: bool = False,
                        gemma4_kv_sharing_map: "list[dict] | None" = None,
                        gemma4_target_kv_cache_quant: "str | None" = None,
                        num_decoder_layers: "int | None" = None) -> nn.Module:
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
            reduced_vocab_dir:
                            Optional directory containing ``vocab_map.safetensors``.
            mtp_base:       When True, export the standard Qwen3.5 text model as
                            the tree-verification MTP base variant.
            mtp_draft:      When True, build the dedicated Qwen3.5 dense MTP
                            draft model from the base checkpoint config. Qwen-style
                            MTP always emits logits for tree verification.
            tp_size:        Tensor-parallel world size.  When >1 the config
                            is reduced to per-rank shapes via
                            :meth:`ModelConfig.for_rank`, and weights
                            are sharded on assignment.  Default 1 = no TP.
            tp_rank:        This rank's index in [0, tp_size).
            mtp_tree_base:  Deprecated compatibility switch. Qwen-style MTP
                            always uses DDTree export semantics.
            dflash_base:    When True, export as DFlash base model.
            dflash_tree_base:
                            When True, add DDTree parent/depth metadata inputs
                            for Qwen3.5 hybrid state execution.
            dflash_draft:   When True, build the DFlash draft model.
            dflash_draft_dir:
                            Path to the DFlash draft checkpoint directory.
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
                            Only supported for the plain default ``CausalLM``
                            path (e.g. Qwen3); rejected for eagle/mtp/dflash and
                            registered non-default variants. The checkpoint's
                            extra-layer weights are simply skipped by the loader.

        Returns:
            Loaded ``nn.Module`` in eval mode.
        """
        from .models.default.modeling_default import CausalLM

        config = load_model_config(model_dir)
        if eagle_base:
            config.eagle_base = True
        if mtp_base or config.mtp_base:
            config.mtp_base = True
            config.mtp_tree_base = True
        if gemma4_mtp_base:
            config.gemma4_mtp_base = True
        if mtp_tree_base or mtp_draft:
            config.mtp_tree_base = True
        if config.mtp_tree_base and not mtp_draft:
            config.mtp_base = True
        if dflash_base:
            config.dflash_base = True
        if dflash_tree_base:
            config.dflash_base = True
            config.dflash_tree_base = True
        elif config.dflash_tree_base:
            config.dflash_base = True
        if config.dflash_base:
            # Read target_layer_ids from DFlash draft checkpoint if provided
            if not config.dflash_target_layer_ids and dflash_draft_dir:
                import json
                draft_cfg_path = os.path.join(dflash_draft_dir, "config.json")
                if os.path.isfile(draft_cfg_path):
                    with open(draft_cfg_path) as f:
                        draft_cfg = json.load(f)
                    dflash_cfg = draft_cfg.get("dflash_config", {})
                    config.dflash_target_layer_ids = dflash_cfg.get(
                        "target_layer_ids", [1, 8, 15, 22, 29])
                    config.dflash_block_size = dflash_cfg.get("block_size", 16)
                    config.dflash_mask_token_id = dflash_cfg.get(
                        "mask_token_id", 248070)
            if not config.dflash_target_layer_ids:
                config.dflash_target_layer_ids = [1, 8, 15, 22, 29]
        if tp_size > 1:
            config = config.for_rank(tp_rank, tp_size)

        variant = _resolve_model_variant(config,
                                         eagle_base=eagle_base,
                                         mtp_base=config.mtp_base,
                                         mtp_draft=mtp_draft,
                                         dflash_base=config.dflash_base,
                                         dflash_draft=dflash_draft,
                                         gemma4_mtp_base=gemma4_mtp_base,
                                         gemma4_mtp_draft=gemma4_mtp_draft)

        # EAGLE3 draft: auto-detect from draft_vocab_size
        if variant == "eagle3_draft":
            from .models.eagle3.modeling_eagle3_draft import Eagle3DraftModel
            model_class = Eagle3DraftModel
            # Set up key remapping: midlayer -> layers.0, skip t2d
            if key_remap is None:
                key_remap = _eagle3_key_remap
        elif variant == "mtp_draft":
            # TODO: support other model types
            if not _is_qwen3_5_mtp_draft_supported(config.model_type):
                raise NotImplementedError(
                    "MTP draft is only supported for qwen3_5_text checkpoints; "
                    f"got {config.model_type!r}.")
            from .models.qwen3_5 import Qwen3_5MtpDraftModel
            tie_word_embeddings = config.tie_word_embeddings
            config = make_mtp_draft_config(config)
            model_class = Qwen3_5MtpDraftModel
            if key_remap is None:

                def key_remap(key):
                    return _mtp_key_remap(
                        key, tie_word_embeddings=tie_word_embeddings)

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
            config = make_dflash_draft_config(dflash_draft_dir,
                                              standard_attention_scale)
            if not draft_has_lm_head:
                config = _inherit_dflash_lm_head_quant(config, base_config)
            model_class = DFlashDraftModel
            model_dir = dflash_draft_dir
            if key_remap is None:
                key_remap = _dflash_key_remap
        elif variant == "gemma4_mtp_draft":
            if config.root_model_type != "gemma4_assistant":
                raise ValueError(
                    "Gemma4 MTP draft requires a gemma4_assistant checkpoint.")
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
        else:
            if (variant == "mtp_base"
                    and not _is_qwen3_5_mtp_base_supported(config.model_type)):
                raise NotImplementedError(
                    "Qwen3.5 MTP base is only supported for qwen3_5_text "
                    "qwen3_5_moe, or qwen3_5_moe_text checkpoints; "
                    f"got {config.model_type!r}.")
            if variant == "gemma4_mtp_base":
                if config.model_type not in ("gemma4", "gemma4_text"):
                    raise ValueError(
                        "Gemma4 MTP base requires a gemma4/gemma4_text target checkpoint."
                    )
                from .models.gemma4 import Gemma4ForCausalLM
                config.gemma4_mtp_base = True
                model_class = Gemma4ForCausalLM
            else:
                # DFlash base is supported for both Qwen3.5 hybrid (qwen3_5_text) and
                # dense Qwen3 (default CausalLM). Dense models use the Transformer's
                # dflash_target_layer_ids parameter to collect target-layer hidden states.
                model_class = _MODEL_REGISTRY.get(config.model_type, CausalLM)

        # 4-layer numeric validation: truncate to the first N decoder
        # layers.  The whole pipeline is config-driven (the Transformer builds
        # layers from ``config.num_hidden_layers`` / ``config.layer_types``,
        # ``onnx_export_spec`` derives the KV / recurrent / conv I/O counts from
        # them, the runtime config.json is written from ``model.config``, and the
        # loader silently skips checkpoint keys for the dropped layers), so a
        # single config override is sufficient and the modeling code needs no
        # change.  This covers the plain default ``CausalLM`` path (e.g. Qwen3)
        # AND registered hybrid base models (Qwen3.5 linear+full / Gated DeltaNet,
        # Nemotron-H Mamba), which also build per-layer from ``layer_types``.  The
        # eagle/mtp/dflash/gemma4-mtp speculative-decoding variants have a
        # different per-layer structure and remain out of scope.
        if num_decoder_layers is not None:
            if (eagle_base or config.eagle_base or mtp_base or config.mtp_base
                    or dflash_base or config.dflash_base or mtp_draft
                    or dflash_draft or gemma4_mtp_base
                    or config.gemma4_mtp_base or gemma4_mtp_draft
                    or config.gemma4_mtp_draft):
                raise NotImplementedError(
                    "num_decoder_layers cannot be combined with the "
                    "eagle/mtp/dflash/gemma4-mtp speculative-decoding variants."
                )
            if not 1 <= num_decoder_layers <= config.num_hidden_layers:
                raise ValueError(
                    f"num_decoder_layers={num_decoder_layers} out of range "
                    f"[1, {config.num_hidden_layers}].")
            config.num_hidden_layers = num_decoder_layers
            # Keep per-layer lists consistent with the truncated layer count.
            if config.layer_types:
                config.layer_types = config.layer_types[:num_decoder_layers]
            logging.getLogger(__name__).info(
                "num_decoder_layers: truncated to first %d decoder layers",
                num_decoder_layers)

        model = model_class(config)
        model.to(device)

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

        if variant == "dflash_draft" and not draft_has_lm_head:
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

        load_weights(model,
                     model_dir,
                     device=device,
                     key_remap=key_remap,
                     key_prefix=key_prefix,
                     pre_repack_hook=pre_repack_hook,
                     mapping=config.mapping)
        if variant == "dflash_draft":
            if draft_has_lm_head:
                logging.getLogger(__name__).info(
                    "DFlash lm_head source: draft checkpoint buffers")
        if apply_reduced_vocab_after_load:
            from .vocab_reduction.onnx_export import \
                apply_reduced_vocab_from_dir
            apply_reduced_vocab_from_dir(model, reduced_vocab_dir)

        # Post-load optimisation: fuse GDN input projections for Qwen3.5 / Qwen3.5-MoE.
        if (config.model_type in ("qwen3_5_text", "qwen3_5_moe_text")
                and not mtp_draft):
            from .models.qwen3_5 import fuse_gdn_input_projections
            fuse_gdn_input_projections(model)

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
    use_base_group_size = lm_head_quant in _GROUP_SIZE_LM_HEAD_QUANTS
    if (use_base_group_size and draft_quant.quant_type != QUANT_FP16
            and draft_quant.group_size != base_quant.group_size):
        raise ValueError(
            "DFlash draft cannot share %s base lm_head with a different "
            "draft quantization group size." % lm_head_quant)

    layer_overrides = dict(draft_quant.layer_overrides)
    layer_overrides["lm_head"] = lm_head_quant
    excluded = [name for name in draft_quant.excluded if name != "lm_head"]
    quant_updates = {
        "excluded": excluded,
        "layer_overrides": layer_overrides,
    }
    if use_base_group_size:
        quant_updates["group_size"] = base_quant.group_size
    if lm_head_quant == QUANT_INT4_GPTQ:
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
            "dflash_base and dflash_draft cannot both be enabled.")
    if dflash_base and (eagle_base or mtp_base or mtp_draft):
        raise ValueError(
            "dflash_base cannot be combined with eagle/mtp variants.")
    if dflash_draft and (eagle_base or mtp_base or mtp_draft):
        raise ValueError(
            "dflash_draft cannot be combined with eagle/mtp variants.")
    if gemma4_mtp_base and (eagle_base or mtp_base or mtp_draft or dflash_base
                            or dflash_draft):
        raise ValueError(
            "gemma4_mtp_base cannot be combined with other speculative variants."
        )
    if gemma4_mtp_draft and (eagle_base or mtp_base or mtp_draft or dflash_base
                             or dflash_draft):
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
        return "eagle3_draft"
    if gemma4_mtp_draft:
        return "gemma4_mtp_draft"
    if dflash_draft:
        return "dflash_draft"
    if dflash_base:
        return "dflash_base"
    if mtp_draft:
        return "mtp_draft"
    if mtp_base:
        return "mtp_base"
    if gemma4_mtp_base:
        return "gemma4_mtp_base"
    if eagle_base:
        return "eagle_base"
    return "llm"


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


def _load_dflash_lm_head(model: nn.Module,
                         base_model_dir: str,
                         device: str,
                         *,
                         tie_word_embeddings: bool = True) -> None:
    """Load the DFlash draft lm_head from the base model checkpoint.

    Deterministic source selection (matching MTP pattern):
      1. Explicit ``lm_head.weight`` from the base checkpoint.
      2. Embedding fallback *only* when ``tie_word_embeddings=True``.
      3. Otherwise fail loudly — untied models must not use embeddings.

    This helper is used only for old DFlash draft checkpoints that do not carry
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
        logger.warning("DFlash draft model has no lm_head; skipping.")
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
        source_key = next((f"{prefix}.{target_name}"
                           for prefix in source_prefixes
                           if f"{prefix}.{target_name}" in shard_map), None)
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


def _checkpoint_has_dflash_lm_head(model_dir: str) -> bool:
    """Return whether a DFlash draft checkpoint owns lm_head tensors."""
    from .checkpoint.loader import _build_shard_map

    for key in _build_shard_map(model_dir):
        mapped = _dflash_key_remap(key)
        if mapped is not None and mapped.startswith("lm_head."):
            return True
    return False


def _dflash_key_remap(key: str) -> "str | None":
    """Remap DFlash draft checkpoint keys."""
    if "rotary_emb" in key:
        return None
    return key


def _mtp_key_remap(key: str, *, tie_word_embeddings: bool) -> "str | None":
    """Remap MTP checkpoint keys for the draft model.

    The embedding table is only a valid LM-head fallback when the source
    checkpoint ties word embeddings.
    """
    if key.startswith("mtp."):
        return key[len("mtp."):]
    if key.startswith("lm_head."):
        return key
    if tie_word_embeddings and key in (
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight"):
        return "lm_head.weight"
    return None
