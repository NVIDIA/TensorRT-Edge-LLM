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
"""Model configuration for the device builder.

Reads ``config.json`` + ``hf_quant_config.json`` (or the embedded
``quantization_config``) with plain ``json`` -- no ``transformers.AutoConfig``
and no framework imports. Component-specific dictionaries remain available
for visual, audio, vocoder, and action builders.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tensorrt_edgellm.dflash import DFlashVersion, resolve_dflash_contract

from . import contracts, quantization
from .bundle import LLM_COMPONENTS, BundleConfig
from .dspark_config import resolve_dspark_config

# Per-layer block type labels.
LAYER_ATTN = "attention"
LAYER_MAMBA = "mamba"
LAYER_MLP = "mlp"
LAYER_MOE = "moe"
LAYER_GDN = "gdn"

_DEFAULT_ROPE_THETA = 10000.0


@dataclass
class MambaConfig:
    num_heads: int
    head_dim: int
    ssm_state_size: int
    conv_dim: int
    conv_kernel: int
    n_groups: int

    @property
    def intermediate_size(self) -> int:
        return self.num_heads * self.head_dim


@dataclass
class GdnConfig:
    """Gated DeltaNet dimensions for hybrid decoder layers."""

    num_key_heads: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_kernel: int

    @property
    def key_dim(self) -> int:
        return self.num_key_heads * self.key_head_dim

    @property
    def value_dim(self) -> int:
        return self.num_value_heads * self.value_head_dim

    @property
    def conv_dim(self) -> int:
        return self.key_dim * 2 + self.value_dim


@dataclass
class DeviceConfig:
    """Flat model configuration consumed by the device builders."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    model_dir: str = ""
    root_model_type: str = ""
    rope_scaling: Optional[dict] = None
    original_max_position_embeddings: Optional[int] = None
    partial_rotary_factor: float = 1.0
    global_head_dim: int = 0
    num_global_key_value_heads: int = 0
    sliding_rope_config: Optional[dict] = None
    full_rope_config: Optional[dict] = None
    attention_layer_types: List[str] = field(default_factory=list)
    attention_bias: bool = False
    attention_k_eq_v: bool = False
    attention_sink_bias: bool = False
    tie_word_embeddings: bool = False
    sliding_window_size: int = -1
    final_logit_softcapping: Optional[float] = None
    skip_softmax_scale_factor: float = 0.0

    # quantization
    quant: quantization.QuantConfig = field(
        default_factory=quantization.QuantConfig)

    # per-layer block types
    layer_types: List[str] = field(default_factory=list)

    # sparse MoE
    num_experts: int = 0
    n_routed_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    moe_shared_expert_intermediate_size: int = 0
    moe_latent_size: Optional[int] = None
    routed_scaling_factor: float = 1.0
    n_group: int = 1
    topk_group: int = 1
    decoder_sparse_step: int = 1
    mlp_only_layers: List[int] = field(default_factory=list)
    norm_topk_prob: bool = True

    # feature flags
    num_deepstack_features: int = 0

    # component and speculative behavior
    component: str = contracts.Component.LLM.value
    engine_role: str = "llm"
    spec_decode_type: str = "none"
    accept_hidden_layer: int = -1
    attention_scaling: float = 1.0
    embedding_scale: float = 1.0
    has_value_norm: bool = False
    attn_output_gate: bool = False
    rotary_dim_override: int = 0
    hybrid_uses_rope: bool = True
    hidden_act: str = "silu"
    mamba_hidden_act: str = "silu"
    tp_size: int = 1
    tp_rank: int = 0

    # speculative decoding
    mtp_num_hidden_layers: Optional[int] = None
    mtp_use_dedicated_embeddings: bool = False
    eagle_base: bool = False
    eagle3_target_layer_ids: List[int] = field(default_factory=list)
    mtp_base: bool = False
    mtp_tree_base: bool = False
    dflash_base: bool = False
    dflash_tree_base: bool = False
    dflash_target_layer_ids: List[int] = field(default_factory=list)
    dflash_block_size: int = 16
    dflash_mask_token_id: int = 248070
    dflash_version: DFlashVersion = DFlashVersion.V1
    dflash2_target_layer_ids: List[int] = field(default_factory=list)
    dflash2_block_size: int = 0
    dflash2_mask_token_id: int = -1
    dflash2_is_causal: bool = True
    dflash2_conv_kernel_size: int = 0
    dflash2_conv_group_size: int = 0
    dflash2_selector_rank: int = 0
    dflash2_selector_top_k: int = 0
    dspark_base: bool = False
    dspark_tree_base: bool = False
    dspark_target_layer_ids: List[int] = field(default_factory=list)
    dspark_block_size: int = 7
    dspark_mask_token_id: int = 151669
    dspark_enable_confidence_head: bool = False
    dspark_confidence_head_with_markov: bool = False
    dspark_markov_head_type: str = ""
    dspark_markov_rank: int = 0
    dspark_causal_proposal: bool = False
    dspark_contiguous_query_swa: bool = False
    dspark_sample_from_anchor: bool = True
    draft_vocab_size: Optional[int] = None
    target_hidden_size: Optional[int] = None

    # shared-KV and assistant runtime contracts
    raw_layer_types: List[str] = field(default_factory=list)
    rope_parameters: Optional[dict] = None
    backbone_hidden_size: int = 0
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
    hidden_size_per_layer_input: int = 0
    vocab_size_per_layer_input: int = 0
    num_kv_shared_layers: int = 0
    use_double_wide_mlp: bool = False
    enable_moe_block: bool = False
    self_conditioning_size: int = 0

    # runtime vocabulary
    reduced_vocab_size: Optional[int] = None

    # Raw dictionaries retain model-specific encoder and decoder fields.
    raw_root: Dict[str, Any] = field(default_factory=dict)
    raw_component: Dict[str, Any] = field(default_factory=dict)

    # hybrid
    mamba_cfg: Optional[MambaConfig] = None
    gdn_cfg: Optional[GdnConfig] = None

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def is_hybrid(self) -> bool:
        return self.mamba_cfg is not None or self.gdn_cfg is not None

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
    def quant_type(self) -> str:
        return self.quant.quant_type

    @property
    def group_size(self) -> int:
        return self.quant.group_size

    @property
    def kv_cache_quant(self) -> Optional[str]:
        return self.quant.kv_cache_quant

    @property
    def excluded(self) -> List[str]:
        return list(self.quant.excluded)

    def module_quant_type(self, module_name: str) -> str:
        """Return the concrete checkpoint precision for one linear."""
        return self.quant.module_type(module_name, self.tie_word_embeddings)

    @property
    def rotary_dim(self) -> int:
        if self.rotary_dim_override > 0:
            return self.rotary_dim_override
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def kv_cache_dtype(self) -> str:
        return "fp8" if self.kv_cache_quant == "fp8" else "fp16"

    @property
    def uses_dual_rope(self) -> bool:
        return (self.sliding_rope_config is not None
                and self.full_rope_config is not None)

    def rope_rotary_dim(self, rope_config: Optional[dict],
                        head_dim: int) -> int:
        """Return the runtime RoPE table width for one attention type."""
        config = rope_config or {}
        scaling = config.get("rope_scaling")
        if isinstance(scaling, dict):
            rope_type = scaling.get("rope_type", scaling.get("type"))
            if rope_type == "proportional":
                return head_dim
        partial = float(
            config.get("partial_rotary_factor", self.partial_rotary_factor))
        return int(head_dim * partial)

    def rope_partial_rotary_dim(self, rope_config: Optional[dict],
                                head_dim: int) -> int:
        """Return the checkpoint's partial RoPE width."""
        config = rope_config or {}
        partial = float(
            config.get("partial_rotary_factor", self.partial_rotary_factor))
        return int(head_dim * partial)

    @property
    def logits_vocab_size(self) -> int:
        return self.reduced_vocab_size or self.vocab_size

    def attention_type(self, layer_index: int) -> str:
        if layer_index < len(self.attention_layer_types):
            return self.attention_layer_types[layer_index]
        return "full_attention"

    def layer_head_dim(self, layer_index: int) -> int:
        if (self.attention_type(layer_index) == "full_attention"
                and self.global_head_dim > 0):
            return self.global_head_dim
        return self.head_dim

    def layer_num_kv_heads(self, layer_index: int) -> int:
        if (self.attention_type(layer_index) == "full_attention"
                and self.attention_k_eq_v
                and self.num_global_key_value_heads > 0):
            return self.num_global_key_value_heads
        return self.num_key_value_heads

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
            cls,
            model_dir: str,
            component: "contracts.Component | str" = contracts.Component.LLM,
            tp_size: int = 1,
            tp_rank: int = 0,
            num_decoder_layers: Optional[int] = None) -> "DeviceConfig":
        bundle = BundleConfig.from_pretrained(model_dir)
        resolved_component = (component
                              if isinstance(component, contracts.Component)
                              else contracts.Component(component))
        from ..models import registry as model_registry
        configuration = model_registry.configuration_module_for(
            bundle.root_model_type)
        weight_conversion = model_registry.weight_conversion_for(
            bundle.root_model_type)
        root = bundle.root
        selected = bundle.component_dict(resolved_component)
        llm = (_promote_llm_dict(selected)
               if resolved_component in LLM_COMPONENTS else selected)
        prepare_text_config = getattr(configuration, "prepare_text_config",
                                      None)
        if prepare_text_config is not None and resolved_component in LLM_COMPONENTS:
            llm = prepare_text_config(llm, root, resolved_component, model_dir)

        model_type = _component_model_type(llm, selected, root,
                                           resolved_component)
        llm = _normalize_layer_count(llm)
        if num_decoder_layers is not None and resolved_component in LLM_COMPONENTS:
            llm = _truncate_layers(llm, num_decoder_layers)
        hidden_size = int(llm["hidden_size"])
        num_attn_heads = int(llm["num_attention_heads"])
        head_dim = int(llm.get("head_dim", hidden_size // num_attn_heads))

        raw_layer_types = _parse_raw_layer_types(llm)
        layer_types = _parse_layer_types(llm)
        attention_layer_types = _parse_attention_layer_types(
            llm, int(llm["num_hidden_layers"]))
        dual_rope = _get_dual_rope_configs(llm)
        mamba_cfg = _parse_mamba_cfg(llm, layer_types, model_dir)
        gdn_cfg = _parse_gdn_cfg(llm, layer_types)
        quant = quantization.parse_quantization(model_dir, root, llm,
                                                weight_conversion)

        num_experts = int(
            llm.get("num_experts", llm.get("num_local_experts", 0)) or 0)
        n_routed_experts = int(llm.get("n_routed_experts", 0) or 0)
        if num_experts == 0 and n_routed_experts > 0:
            num_experts = n_routed_experts
        if n_routed_experts == 0 and num_experts > 0:
            n_routed_experts = num_experts

        intermediate_size = int(
            llm.get("intermediate_size") or llm.get("moe_intermediate_size", 0)
            or 0)

        moe_latent = llm.get("moe_latent_size", None)
        mtp_num_hidden_layers = llm.get("mtp_num_hidden_layers",
                                        llm.get("num_nextn_predict_layers"))

        dflash_contract = resolve_dflash_contract(root, llm)
        dspark_sliding_window = _get_sliding_window(llm)
        dspark = resolve_dspark_config(llm, dspark_sliding_window)
        dspark_sliding_window = int(dspark["sliding_window_size"])
        result = cls(
            model_type=model_type,
            model_dir=model_dir,
            root_model_type=bundle.root_model_type,
            hidden_size=hidden_size,
            num_hidden_layers=int(llm["num_hidden_layers"]),
            num_attention_heads=num_attn_heads,
            num_key_value_heads=int(
                llm.get("num_key_value_heads", num_attn_heads)),
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            vocab_size=int(llm["vocab_size"]),
            rms_norm_eps=_get_rms_norm_eps(llm),
            rope_theta=_get_rope_theta(llm),
            max_position_embeddings=int(
                llm.get("max_position_embeddings", 4096)),
            rope_scaling=_select_rope_scaling(llm),
            original_max_position_embeddings=(
                int(llm["original_max_position_embeddings"])
                if llm.get("original_max_position_embeddings") is not None else
                None),
            partial_rotary_factor=_get_partial_rotary_factor(llm),
            global_head_dim=int(llm.get("global_head_dim", 0) or 0),
            num_global_key_value_heads=int(
                llm.get("num_global_key_value_heads", 0) or 0),
            sliding_rope_config=dual_rope.get("sliding_rope_config"),
            full_rope_config=dual_rope.get("full_rope_config"),
            attention_layer_types=attention_layer_types,
            attention_bias=bool(llm.get("attention_bias", False)),
            attention_k_eq_v=bool(llm.get("attention_k_eq_v", False)),
            attention_sink_bias=bool(dspark["attention_sink_bias"]),
            tie_word_embeddings=bool(llm.get("tie_word_embeddings", False)),
            sliding_window_size=dspark_sliding_window,
            skip_softmax_scale_factor=_get_skip_softmax_scale_factor(
                llm, root),
            final_logit_softcapping=(float(llm["final_logit_softcapping"])
                                     if llm.get("final_logit_softcapping")
                                     is not None else None),
            quant=quant,
            layer_types=layer_types,
            num_experts=num_experts,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=int(
                llm.get("num_experts_per_tok", llm.get("top_k_experts", 0))
                or 0),
            moe_intermediate_size=int(
                llm.get("moe_intermediate_size", 0) or 0),
            moe_shared_expert_intermediate_size=int(
                llm.get("moe_shared_expert_intermediate_size",
                        llm.get("shared_expert_intermediate_size", 0)) or 0),
            moe_latent_size=int(moe_latent)
            if moe_latent is not None else None,
            routed_scaling_factor=float(llm.get("routed_scaling_factor", 1.0)),
            n_group=int(llm.get("n_group", 1)),
            topk_group=int(llm.get("topk_group", 1)),
            decoder_sparse_step=int(llm.get("decoder_sparse_step", 1)),
            mlp_only_layers=list(llm.get("mlp_only_layers") or []),
            norm_topk_prob=bool(llm.get("norm_topk_prob", True)),
            num_deepstack_features=(_parse_num_deepstack_features(llm, root)
                                    if resolved_component
                                    == contracts.Component.LLM else 0),
            component=resolved_component.value,
            accept_hidden_layer=_parse_accept_hidden_layer(llm, root),
            attention_scaling=_get_attention_scaling(llm, head_dim),
            embedding_scale=_get_embedding_scale(llm),
            has_value_norm=_get_has_value_norm(llm),
            attn_output_gate=bool(llm.get("attn_output_gate", False)),
            rotary_dim_override=int(llm.get("rotary_dim_override", 0) or 0),
            hybrid_uses_rope=bool(llm.get("hybrid_uses_rope", True)),
            hidden_act=str(
                llm.get(
                    "mlp_hidden_act",
                    llm.get("hidden_activation", llm.get("hidden_act",
                                                         "silu")))),
            mamba_hidden_act=str(llm.get("mamba_hidden_act", "silu")),
            tp_size=tp_size,
            tp_rank=tp_rank,
            mtp_num_hidden_layers=(int(mtp_num_hidden_layers)
                                   if mtp_num_hidden_layers is not None else
                                   None),
            mtp_use_dedicated_embeddings=bool(
                llm.get("mtp_use_dedicated_embeddings", False)),
            eagle_base=bool(llm.get("eagle_base", False)),
            eagle3_target_layer_ids=list(
                llm.get("eagle3_target_layer_ids",
                        root.get("eagle3_target_layer_ids", [])) or []),
            mtp_base=bool(llm.get("mtp_base", False)),
            mtp_tree_base=bool(llm.get("mtp_tree_base", False)),
            dflash_base=bool(llm.get("dflash_base", False)),
            dflash_tree_base=bool(llm.get("dflash_tree_base", False)),
            dflash_target_layer_ids=list((llm.get("dflash_config") or {}).get(
                "target_layer_ids",
                llm.get("dflash_target_layer_ids",
                        llm.get("target_layer_ids", [])))),
            dflash_block_size=int((llm.get("dflash_config") or {}).get(
                "block_size",
                llm.get("dflash_block_size", llm.get("block_size", 16)))),
            dflash_mask_token_id=int((llm.get("dflash_config") or {}).get(
                "mask_token_id",
                llm.get("dflash_mask_token_id",
                        llm.get("mask_token_id", 248070)))),
            dflash_version=dflash_contract.version,
            dflash2_target_layer_ids=list((llm.get("dflash_config")
                                           or {}).get("target_layer_ids", [])),
            dflash2_block_size=int((llm.get("dflash_config")
                                    or {}).get("block_size", 0)),
            dflash2_mask_token_id=int((llm.get("dflash_config")
                                       or {}).get("mask_token_id", -1)),
            dflash2_is_causal=bool(llm.get("is_causal", True)),
            dflash2_conv_kernel_size=int((llm.get("dflash_config")
                                          or {}).get("conv_kernel_size", 0)),
            dflash2_conv_group_size=int((llm.get("dflash_config")
                                         or {}).get("conv_group_size", 0)),
            dflash2_selector_rank=int((llm.get("dflash_config")
                                       or {}).get("selector_rank", 0)),
            dflash2_selector_top_k=int((llm.get("dflash_config")
                                        or {}).get("selector_top_k", 0)),
            dspark_base=bool(llm.get("dspark_base", False)),
            dspark_tree_base=bool(llm.get("dspark_tree_base", False)),
            dspark_target_layer_ids=list(
                dspark.get(
                    "target_layer_ids",
                    llm.get("dspark_target_layer_ids",
                            llm.get("target_layer_ids", [])))),
            dspark_block_size=int(
                dspark.get(
                    "block_size",
                    llm.get("dspark_block_size", llm.get("block_size", 7)))),
            dspark_mask_token_id=int(
                dspark.get(
                    "mask_token_id",
                    llm.get("dspark_mask_token_id",
                            llm.get("mask_token_id", 151669)))),
            dspark_enable_confidence_head=bool(
                dspark.get("enable_confidence_head",
                           llm.get("enable_confidence_head", False))),
            dspark_confidence_head_with_markov=bool(
                dspark.get("confidence_head_with_markov",
                           llm.get("confidence_head_with_markov", False))),
            dspark_markov_head_type=str(
                dspark.get("markov_head_type", llm.get("markov_head_type",
                                                       ""))),
            dspark_markov_rank=int(
                dspark.get("markov_rank", llm.get("markov_rank", 0)) or 0),
            dspark_causal_proposal=bool(
                dspark.get(
                    "causal_head",
                    dspark.get(
                        "causal",
                        llm.get("causal_head",
                                llm.get("dflash_query_causal", False))))),
            dspark_contiguous_query_swa=bool(dspark["contiguous_query_swa"]),
            dspark_sample_from_anchor=bool(
                dspark.get("sample_from_anchor",
                           llm.get("sample_from_anchor", True))),
            draft_vocab_size=(int(llm["draft_vocab_size"])
                              if llm.get("draft_vocab_size") is not None else
                              None),
            target_hidden_size=(int(llm["target_hidden_size"])
                                if llm.get("target_hidden_size") is not None
                                else None),
            raw_layer_types=raw_layer_types,
            rope_parameters=(dict(llm["rope_parameters"]) if isinstance(
                llm.get("rope_parameters"), dict) else None),
            backbone_hidden_size=int(
                llm.get("backbone_hidden_size",
                        root.get("backbone_hidden_size", 0)) or 0),
            assistant_hidden_size=int(
                llm.get("assistant_hidden_size",
                        root.get("assistant_hidden_size", 0)) or 0),
            shares_target_kv=bool(
                llm.get("shares_target_kv", root.get("shares_target_kv",
                                                     False))),
            has_own_kv_cache=bool(
                llm.get("has_own_kv_cache", root.get("has_own_kv_cache",
                                                     True))),
            constant_draft_positions=bool(
                llm.get("constant_draft_positions",
                        root.get("constant_draft_positions", False))),
            returns_feedback_hidden=bool(
                llm.get("returns_feedback_hidden",
                        root.get("returns_feedback_hidden", False))),
            use_ordered_embeddings=bool(
                llm.get("use_ordered_embeddings",
                        root.get("use_ordered_embeddings", False))),
            num_centroids=int(
                llm.get("num_centroids", root.get("num_centroids", 0)) or 0),
            centroid_intermediate_top_k=int(
                llm.get("centroid_intermediate_top_k",
                        root.get("centroid_intermediate_top_k", 0)) or 0),
            sparse_logits_enabled=bool(
                llm.get("sparse_logits_enabled",
                        root.get("sparse_logits_enabled", False))),
            kv_sharing_map=list(llm.get("kv_sharing_map") or []),
            hidden_size_per_layer_input=int(
                llm.get("hidden_size_per_layer_input", 0) or 0),
            vocab_size_per_layer_input=int(
                llm.get("vocab_size_per_layer_input", 0) or 0),
            num_kv_shared_layers=int(llm.get("num_kv_shared_layers", 0) or 0),
            use_double_wide_mlp=bool(llm.get("use_double_wide_mlp", False)),
            enable_moe_block=bool(llm.get("enable_moe_block", False)),
            self_conditioning_size=int(
                llm.get("self_conditioning_size",
                        root.get("self_conditioning_size", intermediate_size))
                or intermediate_size),
            raw_root=root,
            raw_component=selected,
            mamba_cfg=mamba_cfg,
            gdn_cfg=gdn_cfg,
        )
        update_device_config = getattr(configuration, "update_device_config",
                                       None)
        if update_device_config is not None:
            update_device_config(result, root, resolved_component)
        if tp_size > 1:
            dimensions = {
                "num_attention_heads": result.num_attention_heads,
                "num_key_value_heads": result.num_key_value_heads,
                "intermediate_size": result.intermediate_size,
            }
            if result.num_global_key_value_heads:
                dimensions[
                    "num_global_key_value_heads"] = result.num_global_key_value_heads
            invalid = {
                name: value
                for name, value in dimensions.items()
                if value and value % tp_size
            }
            if invalid:
                details = ", ".join(f"{name}={value}"
                                    for name, value in invalid.items())
                raise ValueError(
                    f"TP size {tp_size} does not divide model dimensions: {details}"
                )
            result.num_attention_heads //= tp_size
            result.num_key_value_heads //= tp_size
            result.intermediate_size //= tp_size
            if result.num_global_key_value_heads:
                result.num_global_key_value_heads //= tp_size
            if result.mamba_cfg is not None:
                if result.mamba_cfg.num_heads % tp_size:
                    raise ValueError("TP size must divide mamba_num_heads")
                result.mamba_cfg.num_heads //= tp_size
                result.mamba_cfg.conv_dim //= tp_size
                result.mamba_cfg.n_groups = max(
                    1, result.mamba_cfg.n_groups // tp_size)
            if result.gdn_cfg is not None:
                if (result.gdn_cfg.num_key_heads % tp_size
                        or result.gdn_cfg.num_value_heads % tp_size):
                    raise ValueError("TP size must divide GDN head counts")
                result.gdn_cfg.num_key_heads //= tp_size
                result.gdn_cfg.num_value_heads //= tp_size
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _promote_llm_dict(root: Dict[str, Any]) -> Dict[str, Any]:
    """Return the dict that holds LLM architecture fields."""
    if root.get("num_attention_heads") is not None:
        return root
    for name in ("text_config", "llm_config", "language_config"):
        sub = root.get(name)
        if isinstance(sub,
                      dict) and sub.get("num_attention_heads") is not None:
            return sub
    supplemental = root.get("_direct_vlm_config")
    if isinstance(supplemental, dict):
        promoted = _promote_llm_dict(supplemental)
        if promoted is not supplemental or promoted.get(
                "num_attention_heads") is not None:
            promoted = dict(promoted)
            if root.get("vocab_size") is not None:
                promoted["vocab_size"] = root["vocab_size"]
            return promoted
    return root


def _component_model_type(llm: Dict[str, Any], selected: Dict[str, Any],
                          root: Dict[str, Any],
                          component: contracts.Component) -> str:
    """Resolve the concrete HF architecture represented by a component."""
    if llm.get("model_type"):
        return str(llm["model_type"])
    if component == contracts.Component.TALKER:
        outer_type = str(selected.get("model_type", ""))
        if outer_type.endswith("_talker"):
            return outer_type + "_text"
    root_model_type = root.get("model_type")
    if not isinstance(root_model_type, str) or not root_model_type:
        raise ValueError("config.json must define a non-empty model_type")
    return root_model_type


def _get_rope_theta(llm: Dict[str, Any]) -> float:
    if llm.get("rope_theta") is not None:
        return float(llm["rope_theta"])
    for key in ("rope_scaling", "rope_parameters"):
        nested = llm.get(key)
        if not isinstance(nested, dict):
            continue
        if nested.get("rope_theta") is not None:
            return float(nested["rope_theta"])
        for attention_type in ("full_attention", "sliding_attention"):
            attention_rope = nested.get(attention_type)
            if (isinstance(attention_rope, dict)
                    and attention_rope.get("rope_theta") is not None):
                return float(attention_rope["rope_theta"])
    return _DEFAULT_ROPE_THETA


def _get_partial_rotary_factor(llm: Dict[str, Any]) -> float:
    prf = llm.get("partial_rotary_factor")
    if prf is not None:
        return float(prf)
    for key in ("rope_parameters", "rope_scaling"):
        nested = llm.get(key)
        if isinstance(
                nested,
                dict) and nested.get("partial_rotary_factor") is not None:
            return float(nested["partial_rotary_factor"])
    return 1.0


def _select_rope_scaling(llm: Dict[str, Any]) -> Optional[dict]:
    for key in ("rope_scaling", "rope_parameters"):
        nested = llm.get(key)
        if isinstance(nested, dict):
            if isinstance(nested.get("full_attention"), dict):
                nested = nested["full_attention"]
            elif isinstance(nested.get("sliding_attention"), dict):
                nested = nested["sliding_attention"]
            out = dict(nested)
            rope_type = out.get("rope_type", out.get("type"))
            if rope_type is not None:
                out.setdefault("rope_type", rope_type)
                out.setdefault("type", rope_type)
            return out
    return None


def _get_rms_norm_eps(llm: Dict[str, Any]) -> float:
    return float(
        llm.get("rms_norm_eps",
                llm.get("norm_eps", llm.get("layer_norm_epsilon", 1e-6))))


def _get_attention_scaling(llm: Dict[str, Any], head_dim: int) -> float:
    for key in ("attention_scaling", "qk_scale", "scaling"):
        if llm.get(key) is not None:
            return float(llm[key])
    return 1.0 / math.sqrt(float(head_dim))


def _get_embedding_scale(llm: Dict[str, Any]) -> float:
    for key in ("embedding_scale", "embed_scale", "scalar_embed_scale"):
        if llm.get(key) is not None:
            return float(llm[key])
    return 1.0


def _get_has_value_norm(llm: Dict[str, Any]) -> bool:
    for key in ("has_value_norm", "has_v_norm", "value_norm"):
        if llm.get(key) is not None:
            return bool(llm[key])
    return False


def _get_sliding_window(llm: Dict[str, Any]) -> int:
    use_sw = bool(llm.get("use_sliding_window", False))
    if not use_sw and "sliding_attention" in (llm.get("layer_types") or []):
        use_sw = True
    sw = llm.get("sliding_window") if use_sw else None
    if sw is not None and llm.get("use_bidirectional_attention") == "all":
        sw = int(sw) // 2 + 1
    return int(sw) if sw is not None else -1


def _get_skip_softmax_scale_factor(llm: Dict[str, Any],
                                   root: Dict[str, Any]) -> float:
    """Return the calibrated skip-softmax S (0 = disabled).

    An explicit ``skip_softmax_scale_factor`` wins; otherwise
    ``skip_softmax_target_sparsity`` converts through the recorded
    ``skip_softmax_calibration`` formula ``S = a * exp(b * target_sparsity)``.
    """
    for cfg in (llm, root):
        value = cfg.get("skip_softmax_scale_factor")
        if value is not None:
            return float(value)
    for cfg in (llm, root):
        sparsity = cfg.get("skip_softmax_target_sparsity")
        calibration = cfg.get("skip_softmax_calibration")
        if sparsity is None:
            continue
        if not isinstance(calibration, dict):
            raise ValueError("skip_softmax_target_sparsity requires "
                             "skip_softmax_calibration {a, b} in config.json")
        return float(calibration["a"]) * math.exp(
            float(calibration["b"]) * float(sparsity))
    return 0.0


def _parse_raw_layer_types(llm: Dict[str, Any]) -> List[str]:
    raw = llm.get("layers_block_type") or llm.get("layer_types") or []
    return [str(layer_type) for layer_type in raw]


def _normalize_layer_count(llm: Dict[str, Any]) -> Dict[str, Any]:
    """Honor provider layer schedules that replace ``num_hidden_layers``."""
    raw = llm.get("layers_block_type") or llm.get("layer_types")
    if not isinstance(raw, (list, tuple)) or not raw:
        return llm
    if llm.get("num_hidden_layers") is not None:
        return llm
    normalized = dict(llm)
    normalized["num_hidden_layers"] = len(raw)
    return normalized


def _truncate_layers(llm: Dict[str, Any], num_layers: int) -> Dict[str, Any]:
    """Keep only the first ``num_layers`` decoder layers (few-layer validation).

    Everything downstream is derived from ``num_hidden_layers`` and the per-layer
    type list, and the weight loader pulls tensors by name, so the dropped layers
    are simply never asked for.
    """
    total = int(llm["num_hidden_layers"])
    if not 1 <= num_layers <= total:
        raise ValueError(
            f"num_decoder_layers={num_layers} out of range [1, {total}]")
    truncated = dict(llm)
    truncated["num_hidden_layers"] = num_layers
    for key in ("layers_block_type", "layer_types"):
        raw = truncated.get(key)
        if isinstance(raw, (list, tuple)) and raw:
            truncated[key] = list(raw)[:num_layers]
    return truncated


def _parse_num_deepstack_features(llm: Dict[str, Any], root: Dict[str,
                                                                  Any]) -> int:
    explicit = llm.get("num_deepstack_features",
                       root.get("num_deepstack_features"))
    if explicit is not None:
        return int(explicit)
    thinker = root.get("thinker_config") or {}
    visual = root.get("vision_config") or thinker.get("vision_config") or {}
    indices = visual.get("deepstack_visual_indexes")
    if isinstance(indices, list) and indices:
        return len(indices)
    return 0


def _parse_accept_hidden_layer(llm: Dict[str, Any], root: Dict[str,
                                                               Any]) -> int:
    if llm.get("accept_hidden_layer") is not None:
        return int(llm["accept_hidden_layer"])
    talker = root.get("talker_config") or {}
    if talker.get("accept_hidden_layer") is not None:
        return int(talker["accept_hidden_layer"])
    return int(root.get("accept_hidden_layer", -1))


def _parse_attention_layer_types(llm: Dict[str, Any],
                                 num_layers: int) -> List[str]:
    raw = _parse_raw_layer_types(llm)
    attention_types = [
        layer_type for layer_type in raw
        if layer_type in ("full_attention", "sliding_attention")
    ]
    if len(attention_types) == num_layers:
        result = attention_types
    elif bool(llm.get("use_sliding_window", False)):
        first_sliding_layer = int(llm.get("max_window_layers", num_layers))
        result = [
            "sliding_attention"
            if index >= first_sliding_layer else "full_attention"
            for index in range(num_layers)
        ]
    else:
        result = ["full_attention"] * num_layers
    return result


def _get_dual_rope_configs(llm: Dict[str, Any]) -> Dict[str, dict]:
    rope = llm.get("rope_parameters")
    if not isinstance(rope, dict):
        return {}
    sliding = rope.get("sliding_attention")
    full = rope.get("full_attention")
    if not isinstance(sliding, dict) or not isinstance(full, dict):
        return {}

    def runtime_config(params: dict) -> dict:
        scaling = dict(params)
        rope_type = scaling.get("rope_type", scaling.get("type"))
        if rope_type is not None:
            scaling.setdefault("rope_type", rope_type)
            scaling.setdefault("type", rope_type)
        return {
            "rope_theta":
            float(
                params.get("rope_theta",
                           llm.get("rope_theta", _DEFAULT_ROPE_THETA))),
            "rope_scaling":
            scaling,
            "partial_rotary_factor":
            float(
                params.get("partial_rotary_factor",
                           llm.get("partial_rotary_factor", 1.0))),
            "max_position_embeddings":
            int(llm.get("max_position_embeddings", 4096)),
        }

    return {
        "sliding_rope_config": runtime_config(sliding),
        "full_rope_config": runtime_config(full),
    }


def _parse_layer_types(llm: Dict[str, Any]) -> List[str]:
    raw = llm.get("layers_block_type") or llm.get("layer_types")
    if raw is not None:
        out: List[str] = []
        for bt in raw:
            b = str(bt).lower()
            if "mamba" in b:
                out.append(LAYER_MAMBA)
            elif "gdn" in b or "linear_attention" in b:
                out.append(LAYER_GDN)
            elif b == "moe":
                out.append(LAYER_MOE)
            elif "mlp" in b:
                out.append(LAYER_MLP)
            else:
                out.append(LAYER_ATTN)
        return out
    return [LAYER_ATTN] * int(llm["num_hidden_layers"])


def _parse_mamba_cfg(llm: Dict[str, Any], layer_types: List[str],
                     model_dir: str) -> Optional[MambaConfig]:
    if LAYER_MAMBA not in layer_types:
        return None
    num_heads = int(llm.get("mamba_num_heads", 0))
    head_dim = int(llm.get("mamba_head_dim", 0))
    ssm_state_size = int(llm.get("ssm_state_size", 0))
    conv_kernel = int(llm.get("conv_kernel", llm.get("mamba_d_conv", 4)))
    d_inner = num_heads * head_dim
    n_groups = int(
        llm.get("n_groups",
                llm.get("mamba_n_groups", llm.get("mamba_num_groups", 1))))
    if "conv_dim" in llm:
        conv_dim = int(llm["conv_dim"])
    else:
        detected = _detect_mamba_conv_dim(model_dir)
        conv_dim = detected if detected > 0 else (
            d_inner + 2 * n_groups * ssm_state_size)
    if n_groups == 1 and conv_dim > d_inner and ssm_state_size > 0:
        derived = (conv_dim - d_inner) // (2 * ssm_state_size)
        n_groups = derived if derived > 0 else 1
    return MambaConfig(num_heads, head_dim, ssm_state_size, conv_dim,
                       conv_kernel, n_groups)


def _parse_gdn_cfg(llm: Dict[str, Any],
                   layer_types: List[str]) -> Optional[GdnConfig]:
    if LAYER_GDN not in layer_types:
        return None
    key_heads = int(llm.get("linear_num_key_heads", 0))
    value_heads = int(llm.get("linear_num_value_heads", key_heads))
    key_head_dim = int(llm.get("linear_key_head_dim", 0))
    value_head_dim = int(llm.get("linear_value_head_dim", key_head_dim))
    if not all((key_heads, value_heads, key_head_dim, value_head_dim)):
        raise ValueError(
            "Gated DeltaNet dimensions are incomplete in config.json")
    return GdnConfig(
        num_key_heads=key_heads,
        num_value_heads=value_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        conv_kernel=int(llm.get("linear_conv_kernel_dim", 4)),
    )


def _detect_mamba_conv_dim(model_dir: str) -> int:
    try:
        from .safetensors_np import SafetensorsStore
        store = SafetensorsStore(model_dir)
        try:
            for k in store.keys():
                if k.endswith(".mixer.conv1d.weight"):
                    return int(store.shape(k)[0])
        finally:
            store.close()
    except Exception:
        pass
    return 0
