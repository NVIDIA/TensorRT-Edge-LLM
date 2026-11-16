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
"""Checkpoint config shared by the LiLiCorr Python unit tests."""

# nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr @ abd47806 config.json
LILICORR_CONFIG = {
    "architectures": ["LiLiCorrDraftModel"],
    "model_type": "qwen3",
    "block_size": 8,
    "dflash_config": {
        "mask_token_id": 990,
        "target_layer_ids": [1, 18, 35, 51, 68, 85],
        "use_swa": True,
        "swa_window_size": 1024,
        "causal": False,
        "attention_sink_bias": True,
        "projector_type": "lilicorr",
        "lilicorr_enabled": True,
        "lilicorr_candidate_topk": 8,
        "lilicorr_hidden_size": 1024,
        "lilicorr_num_layers": 2,
        "lilicorr_num_heads": 8,
        "lilicorr_mlp_ratio": 2.0,
        "lilicorr_factor_dim": 1024,
        "lilicorr_vector_eps": 0.0001,
        "lilicorr_logit_scale": 8.0,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
    },
    "hidden_size": 4096,
    "num_hidden_layers": 5,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "intermediate_size": 12288,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-06,
    "vocab_size": 131072,
    "max_position_embeddings": 262144,
    "initializer_range": 0.02,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "num_target_layers": 0,
    "layer_types": ["sliding_attention"] * 5,
    "sliding_window": 1024,
    "attention_sink_bias": True,
    "use_sliding_window": True,
    "max_window_layers": 0,
    "quantization_config": {
        "quant_algo":
        "W4A16_NVFP4",
        "kv_cache_quant_algo":
        None,
        "group_size":
        16,
        "exclude_modules": [
            "*attention_conv*", "*mlp_conv*", "*self_attn*", "*norm",
            "*lilicorr.anchor_out_head", "*lilicorr.factor_input_proj",
            "lilicorr.feature_mlp*", "*lilicorr.feature_mlp.up_proj",
            "*lilicorr.feature_mlp.down_proj", "*lilicorr.in_head",
            "*lilicorr.out_head", "*qkv_proj", "*lm_head"
        ],
        "quant_method":
        "modelopt_fp4",
        "ignore": [
            "*attention_conv*", "*mlp_conv*", "*self_attn*", "*norm",
            "*lilicorr.anchor_out_head", "*lilicorr.factor_input_proj",
            "lilicorr.feature_mlp*", "*lilicorr.feature_mlp.up_proj",
            "*lilicorr.feature_mlp.down_proj", "*lilicorr.in_head",
            "*lilicorr.out_head", "*qkv_proj", "*lm_head"
        ],
    },
}
