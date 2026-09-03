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
"""Shared checkpoint rules for fused MoE weights."""

from typing import List, Optional, Tuple

import torch

SplitWeights = List[Tuple[str, torch.Tensor]]


def split_fused_moe_experts(key: str,
                            tensor: torch.Tensor) -> Optional[SplitWeights]:
    """Expand a fused routed-expert tensor into per-expert weights."""
    if key.endswith(".mlp.experts.gate_up_proj"):
        if tensor.dim() != 3 or tensor.shape[1] % 2:
            raise RuntimeError(
                f"Fused MoE tensor {key!r} has shape {tuple(tensor.shape)}; "
                "expected [num_experts, 2 * intermediate_size, hidden_size].")
        prefix = key[:-len("gate_up_proj")]
        intermediate_size = tensor.shape[1] // 2
        split: SplitWeights = []
        for expert in range(tensor.shape[0]):
            split.extend([
                (f"{prefix}{expert}.gate_proj.weight",
                 tensor[expert, :intermediate_size]),
                (f"{prefix}{expert}.up_proj.weight",
                 tensor[expert, intermediate_size:]),
            ])
        return split

    if key.endswith(".mlp.experts.down_proj"):
        if tensor.dim() != 3:
            raise RuntimeError(
                f"Fused MoE tensor {key!r} has shape {tuple(tensor.shape)}; "
                "expected [num_experts, hidden_size, intermediate_size].")
        prefix = key[:-len("down_proj")]
        return [(f"{prefix}{expert}.down_proj.weight", tensor[expert])
                for expert in range(tensor.shape[0])]

    return None


def split_fused_shared_expert(
        key: str, tensor: torch.Tensor, hidden_size: int,
        intermediate_size: int) -> Optional[SplitWeights]:
    """Expand a fused shared-expert gate/up tensor after validating its shape."""
    if not key.endswith(".mlp.shared_expert.gate_up_proj"):
        return None

    expected = (2 * intermediate_size, hidden_size)
    if tuple(tensor.shape) != expected:
        raise RuntimeError(f"Fused shared-expert tensor {key!r} has shape "
                           f"{tuple(tensor.shape)}; expected {expected}.")

    prefix = key[:-len("gate_up_proj")]
    return [
        (f"{prefix}gate_proj.weight", tensor[:intermediate_size]),
        (f"{prefix}up_proj.weight", tensor[intermediate_size:]),
    ]
