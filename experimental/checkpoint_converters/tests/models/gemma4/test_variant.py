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
"""Behavioral contracts for Gemma4 variant declarations."""

import copy

import pytest

from experimental.checkpoint_converters.models.gemma4.variant import (
    E2B_IT_VARIANT, AttentionType, validate_hf_config)
from experimental.checkpoint_converters.schema import InvalidMappingPlan


def test_e2b_config_matches_the_pinned_public_text_contract():
    """The E2B spec emits the complete reviewed text-only public config."""
    config = E2B_IT_VARIANT.build_hf_config()
    layer_types = [
        value.value for value in (
            AttentionType.SLIDING,
            AttentionType.SLIDING,
            AttentionType.SLIDING,
            AttentionType.SLIDING,
            AttentionType.FULL,
        ) * 7
    ]

    assert config == {
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
        "text_config": {
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attention_k_eq_v": False,
            "bos_token_id": 2,
            "dtype": "bfloat16",
            "enable_moe_block": False,
            "eos_token_id": 1,
            "expert_intermediate_size": None,
            "final_logit_softcapping": 30.0,
            "global_head_dim": 512,
            "head_dim": 256,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_size": 1536,
            "hidden_size_per_layer_input": 256,
            "initializer_range": 0.02,
            "intermediate_size": 6144,
            "layer_types": layer_types,
            "max_position_embeddings": 131072,
            "model_type": "gemma4_text",
            "num_attention_heads": 8,
            "num_experts": None,
            "num_global_key_value_heads": None,
            "num_hidden_layers": 35,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 20,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-6,
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
            "sliding_window": 512,
            "tie_word_embeddings": True,
            "top_k_experts": None,
            "use_bidirectional_attention": None,
            "use_cache": True,
            "use_double_wide_mlp": True,
            "vocab_size": 262144,
            "vocab_size_per_layer_input": 262144,
        },
        "tie_word_embeddings": True,
        "video_token_id": 258884,
        "vision_soft_tokens_per_image": 280,
    }


def test_e2b_variant_declares_global_heads_and_double_wide_transition():
    """Layer geometry comes from the immutable variant rather than tensor inference."""
    assert E2B_IT_VARIANT.head_dim(13) == 256
    assert E2B_IT_VARIANT.head_dim(14) == 512
    assert E2B_IT_VARIANT.ffn_width(14) == 6144
    assert E2B_IT_VARIANT.ffn_width(15) == 12288
    with pytest.raises(IndexError, match="outside variant"):
        E2B_IT_VARIANT.ffn_width(35)


def test_hf_config_validation_rejects_mutated_geometry():
    """A generated config cannot drift from the declared E2B geometry."""
    config = copy.deepcopy(E2B_IT_VARIANT.build_hf_config())
    config["text_config"]["global_head_dim"] = 256

    with pytest.raises(InvalidMappingPlan, match="global_head_dim"):
        validate_hf_config(config, E2B_IT_VARIANT)
