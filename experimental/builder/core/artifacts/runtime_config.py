# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Runtime configuration artifact builders."""

from importlib import metadata
from pathlib import Path
from typing import Any, Dict

from ...ops.functional.attention import KV_PAGE_SIZE
from .. import contracts
from ..config import LAYER_ATTN, LAYER_GDN, LAYER_MAMBA, DeviceConfig


def normalize_rope_scaling(rope_scaling):
    if not isinstance(rope_scaling, dict):
        return rope_scaling
    out = dict(rope_scaling)
    if "type" not in out and "rope_type" in out:
        out["type"] = out["rope_type"]
    return out


def _edgellm_version() -> str:
    """Resolve the package version without importing ``tensorrt_edgellm``."""
    version_path = (Path(__file__).resolve().parents[4] / "tensorrt_edgellm" /
                    "_version.py")
    if version_path.is_file():
        with version_path.open() as version_file:
            for line in version_file:
                if line.startswith("__version__ = "):
                    return line.split("=", maxsplit=1)[1].strip().strip('"\'')
    try:
        return metadata.version("tensorrt-edgellm")
    except metadata.PackageNotFoundError:
        raise RuntimeError("could not resolve the Edge-LLM version; install "
                           "the package or use a full source checkout")


def build_runtime_config(cfg: DeviceConfig, args) -> Dict[str, Any]:
    """Build the runtime configuration for the supported models."""
    out: Dict[str, Any] = {
        "model": cfg.model_type,
        "spec_decode_type": cfg.spec_decode_type,
        "engine_role": cfg.engine_role,
        "edgellm_version": _edgellm_version(),
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "max_position_embeddings": cfg.max_position_embeddings,
        "rope_theta": cfg.rope_theta,
        "rope_scaling": normalize_rope_scaling(cfg.rope_scaling),
        "partial_rotary_factor": cfg.partial_rotary_factor,
        "num_deepstack_features": cfg.num_deepstack_features,
        "kv_cache_dtype": cfg.kv_cache_dtype,
    }

    if cfg.reduced_vocab_size:
        out["reduced_vocab_size"] = cfg.reduced_vocab_size
    if cfg.original_max_position_embeddings is not None:
        out["original_max_position_embeddings"] = (
            cfg.original_max_position_embeddings)
    if cfg.uses_dual_rope:
        out["sliding_rope_config"] = normalize_rope_scaling(
            cfg.sliding_rope_config)
        out["full_rope_config"] = normalize_rope_scaling(cfg.full_rope_config)
        if (args.resolved_spec_role == contracts.SpecRole.DRAFT
                and args.spec_type == "gemma4_mtp"):
            out["sliding_rotary_dim"] = cfg.rope_partial_rotary_dim(
                cfg.sliding_rope_config, cfg.head_dim)
            out["full_rotary_dim"] = cfg.rope_partial_rotary_dim(
                cfg.full_rope_config, cfg.global_head_dim or cfg.head_dim)

    out["ple_enabled"] = cfg.hidden_size_per_layer_input > 0
    out["num_ple_inputs"] = (cfg.num_hidden_layers
                             if cfg.hidden_size_per_layer_input > 0 else 0)
    out["ple_hidden_size"] = cfg.hidden_size_per_layer_input

    if cfg.global_head_dim and cfg.global_head_dim != cfg.head_dim:
        out["global_head_dim"] = cfg.global_head_dim
        out["layer_types"] = ["attention"] * cfg.num_hidden_layers
        out["kv_layer_configs"] = [{
            "num_kv_heads":
            cfg.layer_num_kv_heads(index),
            "head_dim":
            cfg.layer_head_dim(index),
        } for index in range(cfg.num_hidden_layers)]

    if cfg.num_kv_shared_layers > 0 and not cfg.shares_target_kv:
        out["kv_sharing_donors"] = _gemma_kv_donors(cfg)

    if cfg.is_hybrid and cfg.mamba_cfg is not None:
        mc = cfg.mamba_cfg
        out.update({
            "num_linear_attn_layers": cfg.num_mamba_layers,
            "num_attention_layers": cfg.num_attn_layers,
            "recurrent_state_num_heads": mc.num_heads,
            "recurrent_state_head_dim": mc.head_dim,
            "recurrent_state_size": mc.ssm_state_size,
            "conv_dim": mc.conv_dim,
            "conv_kernel": mc.conv_kernel,
            "use_rope": cfg.num_attn_layers > 0 and cfg.hybrid_uses_rope,
            "recurrent_state_dtype": "fp16",
            "conv_state_dtype": "fp16",
        })
        norm_lt = []
        kv_cfgs = []
        for lt in cfg.layer_types:
            if lt == LAYER_ATTN:
                norm_lt.append("attention")
                kv_cfgs.append({
                    "num_kv_heads": cfg.num_key_value_heads,
                    "head_dim": cfg.head_dim,
                })
            elif lt == LAYER_MAMBA:
                norm_lt.append("mamba")
                kv_cfgs.append(None)
        out["layer_types"] = norm_lt
        out["kv_layer_configs"] = kv_cfgs

    if cfg.is_hybrid and cfg.gdn_cfg is not None:
        gc = cfg.gdn_cfg
        out.update({
            "num_linear_attn_layers": cfg.num_gdn_layers,
            "num_attention_layers": cfg.num_attn_layers,
            "recurrent_state_num_heads": gc.num_value_heads,
            "recurrent_state_head_dim": gc.key_head_dim,
            "recurrent_state_size": gc.value_head_dim,
            "conv_dim": gc.conv_dim,
            "conv_kernel": gc.conv_kernel,
            "use_rope": cfg.num_attn_layers > 0,
            "recurrent_state_dtype": "fp32",
            "conv_state_dtype": "fp16",
        })
        norm_lt = []
        kv_cfgs = []
        for layer_type in cfg.layer_types:
            if layer_type == LAYER_ATTN:
                norm_lt.append("attention")
                kv_cfgs.append({
                    "num_kv_heads": cfg.num_key_value_heads,
                    "head_dim": cfg.head_dim,
                })
            elif layer_type == LAYER_GDN:
                norm_lt.append("mamba")
                kv_cfgs.append(None)
        out["layer_types"] = norm_lt
        out["kv_layer_configs"] = kv_cfgs

    if args.resolved_spec_role == contracts.SpecRole.DRAFT:
        out["draft_vocab_size"] = (cfg.reduced_vocab_size
                                   or cfg.draft_vocab_size or cfg.vocab_size)
        if args.spec_type == "eagle3":
            target_hidden = cfg.target_hidden_size or cfg.hidden_size
            target_layers = cfg.eagle3_target_layer_ids
            if not target_layers:
                raise ValueError(
                    "EAGLE3 draft runtime config requires target-layer IDs")
            out["base_model_hidden_size"] = target_hidden * len(target_layers)
        elif args.spec_type == "mtp":
            out["base_model_hidden_size"] = cfg.hidden_size
        elif args.spec_type in ("dflash", "jetspec"):
            targets = cfg.dflash_target_layer_ids or [1, 8, 15, 22, 29]
            out["base_model_hidden_size"] = len(targets) * cfg.hidden_size
            out["block_size"] = cfg.dflash_block_size
        elif args.spec_type == "dspark":
            out["base_model_hidden_size"] = (len(cfg.dspark_target_layer_ids) *
                                             cfg.hidden_size)
            out["block_size"] = cfg.dspark_block_size
        elif args.spec_type == "gemma4_mtp":
            out.update({
                "model":
                "gemma4_assistant",
                "base_model_hidden_size":
                cfg.backbone_hidden_size,
                "assistant_hidden_size":
                cfg.assistant_hidden_size or cfg.hidden_size,
                "shares_target_kv":
                cfg.shares_target_kv,
                "has_own_kv_cache":
                cfg.has_own_kv_cache,
                "constant_draft_positions":
                cfg.constant_draft_positions,
                "returns_feedback_hidden":
                cfg.returns_feedback_hidden,
                "use_ordered_embeddings":
                cfg.use_ordered_embeddings,
                "num_centroids":
                cfg.num_centroids,
                "centroid_intermediate_top_k":
                cfg.centroid_intermediate_top_k,
                "sparse_logits_enabled":
                cfg.sparse_logits_enabled,
                "sliding_window":
                cfg.sliding_window_size,
                "kv_sharing_map":
                list(cfg.kv_sharing_map),
            })

    if args.spec_type == "dflash":
        out["dflash_config"] = {
            "target_layer_ids": cfg.dflash_target_layer_ids
            or [1, 8, 15, 22, 29],
            "block_size": cfg.dflash_block_size,
            "mask_token_id": cfg.dflash_mask_token_id,
        }
        out["dflash_tree_base"] = cfg.dflash_tree_base
    if args.spec_type == "jetspec":
        out["jetspec_config"] = {
            "target_layer_ids": cfg.dflash_target_layer_ids,
            "block_size": cfg.dflash_block_size,
            "mask_token_id": cfg.dflash_mask_token_id,
            "causal_head": True,
        }
        out["jetspec_tree_base"] = cfg.dflash_tree_base
    if args.spec_type == "mtp":
        out["mtp_tree_base"] = cfg.mtp_tree_base
    if args.spec_type == "dspark":
        out["dspark_config"] = {
            "target_layer_ids": cfg.dspark_target_layer_ids,
            "block_size": cfg.dspark_block_size,
            "mask_token_id": cfg.dspark_mask_token_id,
            "enable_confidence_head": cfg.dspark_enable_confidence_head,
            "confidence_head_with_markov":
            cfg.dspark_confidence_head_with_markov,
            "markov_head_type": cfg.dspark_markov_head_type,
            "markov_rank": cfg.dspark_markov_rank,
            "heads_file": "dspark_heads.safetensors",
            "heads_info_file": "dspark_heads_info.json",
        }
    if cfg.eagle_base:
        out["eagle_hidden_state_layers"] = list(cfg.eagle3_target_layer_ids)

    max_kv_pool_pages = args.max_batch_size * (
        (args.max_kv_cache_capacity + KV_PAGE_SIZE - 1) // KV_PAGE_SIZE)
    out["builder_config"] = {
        "tp_size": args.tp_size,
        "max_input_len": args.max_input_len,
        "spec_draft": args.resolved_spec_role == contracts.SpecRole.DRAFT,
        "spec_base": args.resolved_spec_role == contracts.SpecRole.BASE,
        "max_batch_size": args.max_batch_size,
        "max_lora_rank": args.max_lora_rank,
        "max_kv_cache_capacity": args.max_kv_cache_capacity,
        "max_kv_pool_pages": max_kv_pool_pages,
        "max_verify_tree_size": args.max_verify_tree_size,
        "max_draft_tree_size": args.max_draft_tree_size,
    }
    if args.tp_size > 1:
        dimensions = {
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "intermediate_size": cfg.intermediate_size,
        }
        invalid = {
            name: value
            for name, value in dimensions.items() if value % args.tp_size
        }
        if invalid:
            details = ", ".join(f"{name}={value}"
                                for name, value in sorted(invalid.items()))
            raise ValueError(
                f"TP size {args.tp_size} does not divide runtime dimensions: {details}"
            )
        overrides = {
            name: value // args.tp_size
            for name, value in dimensions.items()
        }
        out["rank_configs"] = [{
            "rank": rank,
            "engine": f"llm_world{args.tp_size}_rank{rank}.engine",
            "config_overrides": dict(overrides),
        } for rank in range(args.tp_size)]
    return out


def _gemma_kv_donors(cfg: DeviceConfig) -> list:
    first_shared = cfg.num_hidden_layers - cfg.num_kv_shared_layers
    donors_by_type = {}
    for index in range(first_shared):
        donors_by_type[cfg.attention_type(index)] = index
    donors = [-1] * cfg.num_hidden_layers
    for index in range(first_shared, cfg.num_hidden_layers):
        donors[index] = donors_by_type[cfg.attention_type(index)]
    return donors


def _visual_runtime_config(bundle) -> Dict[str, Any]:
    root = bundle.root
    visual = dict(bundle.component_dict(contracts.Component.VISUAL))
    runtime_type = bundle.root_model_type
    visual["model_type"] = runtime_type
    result: Dict[str, Any] = {
        "model_type": runtime_type,
        "vision_config": visual,
    }
    thinker = root.get("thinker_config") or {}
    supplemental = root.get("_direct_vlm_config") or {}
    text = dict(
        root.get("text_config") or thinker.get("text_config")
        or supplemental.get("text_config") or root.get("llm_config") or {})
    rope = (text.get("rope_scaling") or text.get("rope_parameters")
            or root.get("rope_scaling") or root.get("rope_parameters"))
    for key in ("vision_start_token_id", "vision_end_token_id",
                "image_token_id", "video_token_id", "vocab_size",
                "rope_theta"):
        for source in (root, thinker, supplemental, text):
            if key in source:
                result[key] = source[key]
                break
    if rope:
        normalized_rope = normalize_rope_scaling(rope)
        result["rope_scaling"] = normalized_rope
        text.setdefault("rope_scaling", normalized_rope)
        if "rope_theta" in normalized_rope:
            text.setdefault("rope_theta", normalized_rope["rope_theta"])
    for key in ("vocab_size", "rope_theta"):
        if key in result:
            text.setdefault(key, result[key])
    if text:
        result["text_config"] = text
    return result


def _audio_runtime_config(bundle) -> Dict[str, Any]:
    root = bundle.root
    audio = dict(bundle.component_dict(contracts.Component.AUDIO))
    runtime_type = bundle.root_model_type
    result: Dict[str, Any] = {
        "model_type": runtime_type,
        "audio_config": audio,
    }
    thinker = root.get("thinker_config") or {}
    for key in ("audio_token_id", "audio_start_token_id", "audio_end_token_id",
                "user_token_id"):
        if key in thinker:
            result[key] = thinker[key]
        elif key in root:
            result[key] = root[key]
    text = thinker.get("text_config") or root.get("text_config") or {}
    if text.get("rope_theta") is not None:
        result["text_config"] = {"rope_theta": text["rope_theta"]}
    return result


def default_component_runtime_config(bundle, component: contracts.Component,
                                     args):
    """Return the generic component runtime config for simple components."""
    if component == contracts.Component.VISUAL:
        result = _visual_runtime_config(bundle)
        result["builder_config"] = {
            "min_image_tokens": args.min_image_tokens,
            "max_image_tokens": args.max_image_tokens,
            "max_image_tokens_per_image": args.max_image_tokens_per_image,
        }
        return result
    if component == contracts.Component.AUDIO:
        result = _audio_runtime_config(bundle)
        result["builder_config"] = {
            "min_time_steps": args.min_time_steps,
            "max_time_steps": args.max_time_steps,
            "min_code_len": args.min_code_len,
            "opt_code_len": args.opt_code_len,
            "max_code_len": args.max_code_len,
        }
        return result
    if component == contracts.Component.CODE2WAV:
        return {
            "model_type": f"{bundle.root_model_type}_code2wav",
            "code2wav_config": dict(bundle.component_dict(component)),
            "builder_config": {
                "min_code_len": args.min_code_len,
                "opt_code_len": args.opt_code_len,
                "max_code_len": args.max_code_len,
            },
        }
    return None
