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
"""Speculative-decoding operations."""

from typing import Sequence

from ..tensor import Tensor
from . import core as F
from ._operation import operation


def hidden_state_feedback(hidden_states,
                          all_hidden_states: Sequence,
                          config,
                          *,
                          allow_eagle3: bool = True):
    """Select the provider-defined hidden states consumed by a draft model."""
    feature_axis = len(hidden_states.shape) - 1
    if (allow_eagle3 and config.spec_decode_type == "eagle3"
            and len(all_hidden_states) >= 5):
        layer_ids = config.eagle3_target_layer_ids or (
            2,
            len(all_hidden_states) // 2,
            len(all_hidden_states) - 3,
        )
        indices = tuple(index - 1 for index in layer_ids)
        if any(index < 0 or index >= len(all_hidden_states)
               for index in indices):
            raise ValueError("EAGLE3 target-layer IDs are out of range")
        return F.concatenate(
            tuple(all_hidden_states[index] for index in indices), feature_axis)
    if config.spec_decode_type in ("dflash", "jetspec", "dspark"):
        if config.spec_decode_type == "dspark":
            indices = config.dspark_target_layer_ids
            algorithm = "DSpark"
        else:
            indices = config.dflash_target_layer_ids or [1, 8, 15, 22, 29]
            algorithm = ("JetSpec"
                         if config.spec_decode_type == "jetspec" else "DFlash")
        if any(index < 0 or index >= len(all_hidden_states)
               for index in indices):
            raise ValueError(f"{algorithm} target layer IDs are out of range")
        selected = [all_hidden_states[index] for index in indices]
        return F.concatenate(tuple(selected), feature_axis)
    return hidden_states


def update_dflash_target_cache(key_delta: Tensor, value_delta: Tensor,
                               past_key_value: Tensor, rope_cos_sin: Tensor,
                               delta_positions: Tensor,
                               delta_token_to_sequence: Tensor,
                               kv_page_table: Tensor) -> Tensor:
    """Write target-hidden K/V deltas into the persistent draft cache."""
    return operation("dflash_target_cache_update", [
        key_delta, value_delta, past_key_value, rope_cos_sin, delta_positions,
        delta_token_to_sequence, kv_page_table
    ])


def dflash2_grouped_dynamic_conv(hidden_states: Tensor,
                                 delta: Tensor,
                                 base_kernel: Tensor,
                                 residual: Tensor = None,
                                 *,
                                 block_size: int,
                                 kernel_size: int,
                                 group_size: int) -> Tensor:
    """Apply one production DFlash2 dynamic grouped depthwise convolution.

    The post-conv form fuses an FP32 residual add and emits FP32; RMSNorm
    remains a native TensorRT graph operation. The pre-conv form emits the
    activation dtype.
    """
    if block_size <= 0:
        raise ValueError("DFlash2 dynamic conv block_size must be positive")
    if kernel_size not in (1, 2, 3, 4):
        raise ValueError("DFlash2 dynamic conv kernel_size must be in [1, 4]")
    if group_size <= 0 or group_size % 2:
        raise ValueError(
            "DFlash2 dynamic conv group_size must be positive and even")
    inputs = [hidden_states, delta, base_kernel]
    if residual is not None:
        inputs.append(residual)
    return operation("dflash2_grouped_dynamic_conv",
                     inputs,
                     block_size=block_size,
                     kernel_size=kernel_size,
                     group_size=group_size,
                     fuse_residual=int(residual is not None))
