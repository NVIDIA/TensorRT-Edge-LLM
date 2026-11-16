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
"""Qualification policy for the SM12x dense NVFP4 CuTeDSL GEMM."""

_SUPPORTED_TP2_MODEL_SIGNATURES = frozenset((
    # (model_type, hidden, layers, head_dim, intermediate, q_heads, kv_heads)
    ("qwen3", 2560, 36, 128, 9728, 32, 8),  # Qwen3-4B
    ("qwen3", 4096, 36, 128, 12288, 32, 8),  # Qwen3-8B
    ("qwen3", 5120, 40, 128, 17408, 40, 8),  # Qwen3-14B
    ("qwen3", 5120, 64, 128, 25600, 64, 8),  # Qwen3-32B
    ("qwen3_vl_text", 4096, 36, 128, 12288, 32, 8),  # Qwen3-VL-8B
    ("qwen3_5_text", 5120, 64, 256, 17408, 24, 4),  # Qwen3.5-27B
))


def supports_sm12x_nvfp4_gemm_allreduce(config) -> bool:
    """Return whether *config* matches a qualified SM12x TP2 model.

    Tensor-parallel model configs carry per-rank head and intermediate
    dimensions, so compare their reconstructed global dimensions. The explicit
    signature allow-list keeps unvalidated model sizes and families on the
    portable TensorRT GEMM plus AllReduce path.
    """
    tp_size = int(getattr(config, "tp_size", 1))
    if tp_size != 2:
        return False
    signature = (
        getattr(config, "model_type", ""),
        int(getattr(config, "hidden_size", 0)),
        int(getattr(config, "num_hidden_layers", 0)),
        int(getattr(config, "head_dim", 0)),
        int(getattr(config, "intermediate_size", 0)) * tp_size,
        int(getattr(config, "num_attention_heads", 0)) * tp_size,
        int(getattr(config, "num_key_value_heads", 0)) * tp_size,
    )
    return signature in _SUPPORTED_TP2_MODEL_SIGNATURES
