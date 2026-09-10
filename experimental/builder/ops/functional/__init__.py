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
"""Unified PyTorch-style operation surface.

Model definitions call every operation through this namespace. TensorRT
native layers and Edge-LLM extension layers deliberately have the same API
level; their lowering mechanism is not part of the model contract.
"""

from ._operation import parameter
from ._operation import supports_operation_attribute as supports
from .attention import KV_PAGE_SIZE, attention, gemma4_attention, vit_attention

# isort: off
from .core import (
    Dimension, apply_multidimensional_rope, apply_rope, batch_token, cast,
    concatenate, constant, convolution, deconvolution, dynamic_lora,
    dynamic_reshape, dynamic_slice, embedding_lookup, empty_sequence,
    fourier_features, fused_nvfp4_gemm_all_reduce, gather_last_tokens,
    gather_nd, kv_cache_update, layer_norm, linear, linear_f32,
    linear_f32_from_weights, linear_from_weights, linear_with_weights, matmul,
    normalization, pad_last_dim, pixel_unshuffle, reduce, reshape, rms_norm,
    rotary_embedding, scaled_dot_product_attention, select, shape_of,
    slice_last_dim, tensor, topk, unwrap)
# isort: on
from .distributed import all_reduce
from .moe import MoeActivation, MoeRouting, fp16_moe, int4_moe, nvfp4_moe
from .recurrent import causal_conv1d, gated_delta_net, update_ssm_state
from .speculative import hidden_state_feedback, update_dflash_target_cache

__all__ = [
    "Dimension",
    "KV_PAGE_SIZE",
    "MoeActivation",
    "MoeRouting",
    "all_reduce",
    "apply_multidimensional_rope",
    "apply_rope",
    "attention",
    "batch_token",
    "cast",
    "causal_conv1d",
    "concatenate",
    "constant",
    "convolution",
    "deconvolution",
    "dynamic_lora",
    "dynamic_reshape",
    "dynamic_slice",
    "embedding_lookup",
    "empty_sequence",
    "fourier_features",
    "fp16_moe",
    "fused_nvfp4_gemm_all_reduce",
    "gated_delta_net",
    "gather_last_tokens",
    "gather_nd",
    "gemma4_attention",
    "hidden_state_feedback",
    "int4_moe",
    "kv_cache_update",
    "layer_norm",
    "linear",
    "linear_f32",
    "linear_f32_from_weights",
    "linear_from_weights",
    "linear_with_weights",
    "matmul",
    "normalization",
    "nvfp4_moe",
    "parameter",
    "pad_last_dim",
    "pixel_unshuffle",
    "reduce",
    "reshape",
    "rms_norm",
    "rotary_embedding",
    "scaled_dot_product_attention",
    "select",
    "shape_of",
    "slice_last_dim",
    "supports",
    "tensor",
    "topk",
    "unwrap",
    "update_dflash_target_cache",
    "update_ssm_state",
    "vit_attention",
]
