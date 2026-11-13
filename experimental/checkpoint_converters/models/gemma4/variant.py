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
"""Declarative Gemma4 text-model geometry and Hugging Face configuration."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from ...schema import InvalidMappingPlan
from ...variant import CheckpointVariant


class AttentionType(str, enum.Enum):
    """Supported Gemma4 attention layer types."""

    SLIDING = "sliding_attention"
    FULL = "full_attention"


@dataclass(frozen=True)
class KvSharingSpec:
    """Trailing shared-KV geometry.

    Attributes:
        first_shared_layer: First layer that reuses a donor cache.
        num_shared_layers: Number of trailing cache-sharing layers.
        local_donor_layer: Donor for trailing sliding-attention layers.
        full_donor_layer: Donor for trailing full-attention layers.
    """

    first_shared_layer: int
    num_shared_layers: int
    local_donor_layer: int | None
    full_donor_layer: int | None


@dataclass(frozen=True)
class PleSpec:
    """Per-layer embedding geometry."""

    hidden_size: int


@dataclass(frozen=True)
class MoeSpec:
    """Routed expert geometry."""

    num_experts: int
    intermediate_size: int
    top_k: int


@dataclass(frozen=True)
class Gemma4VariantSpec(CheckpointVariant):
    """Immutable semantic specification for one supported Gemma4 variant.

    Attributes:
        name: Exact supported variant name.
        dimensions: Shared vocabulary, layer, hidden, attention, and FFN sizes.
        attention_pattern: Exact attention type for every decoder layer.
        kv_sharing: Donor and trailing shared-layer geometry.
        ple: Optional per-layer embedding geometry.
        moe: Optional routed expert geometry.
        hf_fields: Stable scalar fields in the HF text configuration.
    """

    name: str
    dimensions: Mapping[str, int]
    attention_pattern: tuple[AttentionType, ...]
    kv_sharing: KvSharingSpec
    ple: PleSpec | None
    moe: MoeSpec | None
    hf_fields: Mapping[str, Any]

    def num_kv_heads(self, layer_index: int) -> int:
        """Return the local or global KV-head count for one layer."""
        if not 0 <= layer_index < self.dimensions["num_hidden_layers"]:
            raise IndexError(f"layer index outside variant: {layer_index}")
        if self.attention_pattern[layer_index] == AttentionType.FULL:
            return self.dimensions.get("num_global_key_value_heads",
                                       self.dimensions["num_key_value_heads"])
        return self.dimensions["num_key_value_heads"]

    def attention_k_eq_v(self, layer_index: int) -> bool:
        """Return whether a full-attention layer stores one shared K/V weight."""
        if not 0 <= layer_index < self.dimensions["num_hidden_layers"]:
            raise IndexError(f"layer index outside variant: {layer_index}")
        return (self.attention_pattern[layer_index] == AttentionType.FULL
                and bool(self.hf_fields["attention_k_eq_v"]))

    def head_dim(self, layer_index: int) -> int:
        """Return the declared attention head width for one layer.

        Args:
            layer_index: Zero-based decoder layer index.

        Returns:
            Local or global head width.

        Raises:
            IndexError: If ``layer_index`` is outside the model.
        """
        if not 0 <= layer_index < self.dimensions["num_hidden_layers"]:
            raise IndexError(f"layer index outside variant: {layer_index}")
        attention_type = self.attention_pattern[layer_index]
        key = "global_head_dim" if attention_type == AttentionType.FULL else "head_dim"
        return self.dimensions[key]

    def ffn_width(self, layer_index: int) -> int:
        """Return the declared FFN width for one layer.

        Args:
            layer_index: Zero-based decoder layer index.

        Returns:
            Base width for unshared layers and double width for trailing layers.
        """
        if not 0 <= layer_index < self.dimensions["num_hidden_layers"]:
            raise IndexError(f"layer index outside variant: {layer_index}")
        if (self.hf_fields["use_double_wide_mlp"]
                and layer_index >= self.kv_sharing.first_shared_layer):
            return self.dimensions["double_wide_intermediate_size"]
        return self.dimensions["intermediate_size"]

    def build_attention_config(self) -> dict[str, Any]:
        """Build exact per-layer attention and dual-RoPE configuration.

        Returns:
            HF text configuration fields for Gemma4 attention.
        """
        config = {
            "global_head_dim":
            self.dimensions["global_head_dim"],
            "head_dim":
            self.dimensions["head_dim"],
            "layer_types": [value.value for value in self.attention_pattern],
            "num_attention_heads":
            self.dimensions["num_attention_heads"],
            "num_global_key_value_heads":
            self.dimensions.get("num_global_key_value_heads"),
            "num_key_value_heads":
            self.dimensions["num_key_value_heads"],
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1_000_000.0,
                    "rope_type": "proportional",
                },
                "sliding_attention": {
                    "rope_theta": 10_000.0,
                    "rope_type": "default",
                },
            },
            "sliding_window":
            self.dimensions["sliding_window"],
        }
        return config

    def build_ple_config(self) -> dict[str, Any]:
        """Build per-layer embedding and shared-KV configuration.

        Returns:
            HF text configuration fields for PLE and KV sharing.
        """
        ple_hidden_size = self.ple.hidden_size if self.ple is not None else 0
        return {
            "hidden_size_per_layer_input": ple_hidden_size,
            "num_hidden_layers": self.dimensions["num_hidden_layers"],
            "num_kv_shared_layers": self.kv_sharing.num_shared_layers,
            "vocab_size_per_layer_input": self.dimensions["vocab_size"],
        }

    def build_moe_config(self) -> dict[str, Any]:
        """Build routed expert configuration from the optional component."""
        if self.moe is None:
            return {
                "enable_moe_block": False,
                "num_experts": None,
                "top_k_experts": None,
            }
        return {
            "enable_moe_block": True,
            "moe_intermediate_size": self.moe.intermediate_size,
            "num_experts": self.moe.num_experts,
            "top_k_experts": self.moe.top_k,
        }

    def build_ffn_config(self) -> dict[str, Any]:
        """Build exact dense FFN and normalization configuration.

        Returns:
            HF text configuration fields for FFN behavior.
        """
        return {
            "hidden_activation": "gelu_pytorch_tanh",
            "intermediate_size": self.dimensions["intermediate_size"],
            "rms_norm_eps": 1e-6,
            "use_double_wide_mlp": self.hf_fields["use_double_wide_mlp"],
        }

    def build_hf_config(self) -> dict[str, Any]:
        """Assemble the official Gemma4 root and text configuration.

        Returns:
            JSON-compatible HF configuration accepted by current Edge loaders.
        """
        text_config = dict(self.hf_fields)
        text_config.update(self.build_attention_config())
        text_config.update(self.build_ple_config())
        text_config.update(self.build_moe_config())
        text_config.update(self.build_ffn_config())
        config = {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "audio_token_id": 258881,
            "boa_token_id": 256000,
            "boi_token_id": 255999,
            "dtype": "bfloat16",
            "eoa_token_id": 258883,
            "eoa_token_index": 258883,
            "eoi_token_id": 258882,
            "eos_token_id": [1, 106],
            "image_token_id": 258880,
            "initializer_range": 0.02,
            "model_type": "gemma4",
            "text_config": text_config,
            "tie_word_embeddings": True,
            "video_token_id": 258884,
            "vision_soft_tokens_per_image": 280,
        }
        validate_hf_config(config, self)
        return config


def validate_hf_config(config: dict[str, Any],
                       variant: Gemma4VariantSpec) -> None:
    """Validate the assembled Gemma4 configuration contract."""
    text_config = config.get("text_config")
    expected = {
        "model_type":
        "gemma4_text",
        "hidden_size":
        variant.dimensions["hidden_size"],
        "num_hidden_layers":
        variant.dimensions["num_hidden_layers"],
        "num_attention_heads":
        variant.dimensions["num_attention_heads"],
        "num_key_value_heads":
        variant.dimensions["num_key_value_heads"],
        "num_global_key_value_heads":
        variant.dimensions.get("num_global_key_value_heads"),
        "head_dim":
        variant.dimensions["head_dim"],
        "global_head_dim":
        variant.dimensions["global_head_dim"],
        "intermediate_size":
        variant.dimensions["intermediate_size"],
        "hidden_size_per_layer_input":
        variant.ple.hidden_size if variant.ple is not None else 0,
        "num_kv_shared_layers":
        variant.kv_sharing.num_shared_layers,
        "sliding_window":
        variant.dimensions["sliding_window"],
        "vocab_size":
        variant.dimensions["vocab_size"],
        "vocab_size_per_layer_input":
        variant.dimensions["vocab_size"],
        "attention_k_eq_v":
        variant.hf_fields["attention_k_eq_v"],
        "enable_moe_block":
        variant.moe is not None,
        "use_double_wide_mlp":
        variant.hf_fields["use_double_wide_mlp"],
    }
    if variant.moe is not None:
        expected.update({
            "moe_intermediate_size": variant.moe.intermediate_size,
            "num_experts": variant.moe.num_experts,
            "top_k_experts": variant.moe.top_k,
        })
    if config.get("model_type") != "gemma4" or not isinstance(
            text_config, dict):
        raise InvalidMappingPlan("invalid Gemma4 root configuration")
    mismatched = [
        name for name, value in expected.items()
        if text_config.get(name) != value
    ]
    expected_attention = [value.value for value in variant.attention_pattern]
    if text_config.get("layer_types") != expected_attention:
        mismatched.append("layer_types")
    if text_config.get("rope_parameters") != variant.build_attention_config(
    )["rope_parameters"]:
        mismatched.append("rope_parameters")
    if mismatched:
        raise InvalidMappingPlan(
            f"invalid Gemma4 config fields: {sorted(mismatched)}")


_E2B_ATTENTION_PATTERN = (
    AttentionType.SLIDING,
    AttentionType.SLIDING,
    AttentionType.SLIDING,
    AttentionType.SLIDING,
    AttentionType.FULL,
) * 7

E2B_IT_VARIANT = Gemma4VariantSpec(
    name="gemma-4-E2B-it",
    dimensions=MappingProxyType({
        "double_wide_intermediate_size": 12288,
        "global_head_dim": 512,
        "head_dim": 256,
        "hidden_size": 1536,
        "intermediate_size": 6144,
        "num_attention_heads": 8,
        "num_hidden_layers": 35,
        "num_key_value_heads": 1,
        "sliding_window": 512,
        "vocab_size": 262144,
    }),
    attention_pattern=_E2B_ATTENTION_PATTERN,
    kv_sharing=KvSharingSpec(
        first_shared_layer=15,
        num_shared_layers=20,
        local_donor_layer=13,
        full_donor_layer=14,
    ),
    ple=PleSpec(hidden_size=256),
    moe=None,
    hf_fields=MappingProxyType({
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attention_k_eq_v": False,
        "bos_token_id": 2,
        "dtype": "bfloat16",
        "eos_token_id": 1,
        "expert_intermediate_size": None,
        "final_logit_softcapping": 30.0,
        "hidden_size": 1536,
        "initializer_range": 0.02,
        "max_position_embeddings": 131072,
        "model_type": "gemma4_text",
        "pad_token_id": 0,
        "tie_word_embeddings": True,
        "use_bidirectional_attention": None,
        "use_cache": True,
        "use_double_wide_mlp": True,
        "vocab_size": 262144,
    }),
)

_A4B_26B_ATTENTION_PATTERN = (
    AttentionType.SLIDING,
    AttentionType.SLIDING,
    AttentionType.SLIDING,
    AttentionType.SLIDING,
    AttentionType.SLIDING,
    AttentionType.FULL,
) * 5

A4B_26B_IT_VARIANT = Gemma4VariantSpec(
    name="gemma-4-26B-A4B-it",
    dimensions=MappingProxyType({
        "double_wide_intermediate_size": 2112,
        "global_head_dim": 512,
        "head_dim": 256,
        "hidden_size": 2816,
        "intermediate_size": 2112,
        "num_attention_heads": 16,
        "num_global_key_value_heads": 2,
        "num_hidden_layers": 30,
        "num_key_value_heads": 8,
        "sliding_window": 1024,
        "vocab_size": 262144,
    }),
    attention_pattern=_A4B_26B_ATTENTION_PATTERN,
    kv_sharing=KvSharingSpec(
        first_shared_layer=30,
        num_shared_layers=0,
        local_donor_layer=None,
        full_donor_layer=None,
    ),
    ple=None,
    moe=MoeSpec(num_experts=128, intermediate_size=704, top_k=8),
    hf_fields=MappingProxyType({
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attention_k_eq_v": True,
        "bos_token_id": 2,
        "dtype": "bfloat16",
        "eos_token_id": 1,
        "final_logit_softcapping": 30.0,
        "hidden_size": 2816,
        "initializer_range": 0.02,
        "max_position_embeddings": 262144,
        "model_type": "gemma4_text",
        "pad_token_id": 0,
        "tie_word_embeddings": True,
        "use_bidirectional_attention": "vision",
        "use_cache": True,
        "use_double_wide_mlp": False,
        "vocab_size": 262144,
    }),
)

VARIANTS = MappingProxyType({
    E2B_IT_VARIANT.name: E2B_IT_VARIANT,
    A4B_26B_IT_VARIANT.name: A4B_26B_IT_VARIANT,
})
