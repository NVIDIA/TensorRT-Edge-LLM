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
"""Mixture-of-experts operations.

The implementation selects the matching lowering for the target architecture
without exposing that choice to model definitions.
"""

from enum import IntEnum

import numpy as np

from ...core.weights import ParameterSpec
from ..tensor import Tensor
from ._operation import operation, parameter

__all__ = [
    "MoeActivation",
    "MoeRouting",
    "fp16_moe",
    "int4_moe",
    "nvfp4_a16_moe",
    "nvfp4_moe",
]


class MoeActivation(IntEnum):
    """Supported fused expert activations."""

    SWIGLU = 2
    RELU2 = 4
    GEGLU = 5


class MoeRouting(IntEnum):
    """Supported fused expert routing modes."""

    SOFTMAX_TOPK = 0
    SIGMOID_GROUP_TOPK = 1
    SOFTMAX_TOPK_POST_SCALE = 2


_BACKEND_AUTO = 0
_IO_DTYPE_FP16 = 1
_MAX_ROUTED_ROWS_AUTO = 0


def _parameter_value(value, dtype):
    if isinstance(value, ParameterSpec):
        if value.dtype != np.dtype(dtype):
            raise TypeError(
                f"parameter metadata dtype {value.dtype} is not {np.dtype(dtype)}"
            )
        return value
    return np.ascontiguousarray(value, dtype=dtype)


def _validate_token_layout(router_logits: Tensor,
                           hidden_states: Tensor) -> None:
    if router_logits.ndim != 2:
        raise ValueError("router_logits must have shape [T, E]")
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [T, H]")


def fp16_moe(router_logits: Tensor,
             hidden_states: Tensor,
             weights: dict,
             num_experts: int,
             top_k: int,
             hidden_size: int,
             moe_inter_size: int,
             *,
             weight_prefix: str,
             weight_bindings: dict,
             norm_topk_prob: int = 1,
             activation_type: MoeActivation = MoeActivation.SWIGLU,
             routing_mode: MoeRouting = MoeRouting.SOFTMAX_TOPK,
             n_group: int = 1,
             topk_group: int = 1,
             routed_scaling_factor: float = 1.0,
             e_score_correction_bias: Tensor | None = None) -> Tensor:
    """Run the FP16 grouped-GEMM MoE implementation."""
    _validate_token_layout(router_logits, hidden_states)
    inputs = [
        router_logits,
        hidden_states,
        parameter(weight_prefix + ".fc1_weights",
                  _parameter_value(weights["fc1_weights"], np.float16),
                  "fp16",
                  recipe=weight_bindings["fc1_weights"]),
        parameter(weight_prefix + ".fc2_weights",
                  _parameter_value(weights["fc2_weights"], np.float16),
                  "fp16",
                  recipe=weight_bindings["fc2_weights"]),
    ]
    if e_score_correction_bias is not None:
        inputs.append(e_score_correction_bias)
    return operation("fp16_moe",
                     inputs,
                     num_experts=num_experts,
                     top_k=top_k,
                     hidden_size=hidden_size,
                     moe_inter_size=moe_inter_size,
                     activation_type=int(activation_type),
                     routing_mode=int(routing_mode),
                     n_group=n_group,
                     topk_group=topk_group,
                     norm_topk_prob=norm_topk_prob,
                     routed_scaling_factor=routed_scaling_factor,
                     max_routed_rows=_MAX_ROUTED_ROWS_AUTO)


def int4_moe(router_logits: Tensor,
             hidden_states: Tensor,
             weights: dict,
             num_experts: int,
             top_k: int,
             hidden_size: int,
             moe_inter_size: int,
             group_size: int,
             *,
             weight_prefix: str,
             weight_bindings: dict,
             zero_point_offset: int = 1) -> Tensor:
    """Run GPTQ-Marlin mixture-of-experts."""
    _validate_token_layout(router_logits, hidden_states)

    inputs = [
        router_logits,
        hidden_states,
        parameter(weight_prefix + ".fc_gate_up_qweights",
                  _parameter_value(weights["fc_gate_up_qweights"], np.int8),
                  "int4_moe",
                  recipe=weight_bindings["fc_gate_up_qweights"]),
        parameter(weight_prefix + ".fc_gate_up_scales",
                  _parameter_value(weights["fc_gate_up_scales"], np.float16),
                  "int4_moe",
                  recipe=weight_bindings["fc_gate_up_scales"]),
        parameter(weight_prefix + ".fc_down_qweights",
                  _parameter_value(weights["fc_down_qweights"], np.int8),
                  "int4_moe",
                  recipe=weight_bindings["fc_down_qweights"]),
        parameter(weight_prefix + ".fc_down_scales",
                  _parameter_value(weights["fc_down_scales"], np.float16),
                  "int4_moe",
                  recipe=weight_bindings["fc_down_scales"]),
    ]
    return operation("int4_moe",
                     inputs,
                     num_experts=num_experts,
                     top_k=top_k,
                     hidden_size=hidden_size,
                     moe_inter_size=moe_inter_size,
                     activation_type=0,
                     quantization_group_size=group_size)


def nvfp4_a16_moe(router_logits: Tensor, hidden_states: Tensor,
                  moe_weights: dict, num_experts: int, top_k: int,
                  hidden_size: int, moe_inter_size: int,
                  activation_type: MoeActivation, n_group: int,
                  topk_group: int, norm_topk_prob: int,
                  routed_scaling_factor: float, routing_mode: MoeRouting, *,
                  weight_prefix: str) -> Tensor:
    """Run weight-only NVFP4 MoE with FP16 activations."""
    _validate_token_layout(router_logits, hidden_states)
    order = [
        ("fc1_qweights", np.int8),
        ("fc1_block_scales", np.int8),
        ("fc1_global_scales", np.float16),
        ("fc2_qweights", np.int8),
        ("fc2_block_scales", np.int8),
        ("fc2_global_scales", np.float16),
        ("e_score_correction_bias", np.float32),
    ]
    weight_inputs = [
        parameter(weight_prefix + "." + name,
                  _parameter_value(moe_weights[name], dtype), "nvfp4_a16_moe")
        for name, dtype in order
    ]
    return operation("nvfp4_a16_moe",
                     [router_logits, hidden_states] + weight_inputs,
                     num_experts=num_experts,
                     top_k=top_k,
                     hidden_size=hidden_size,
                     moe_inter_size=moe_inter_size,
                     activation_type=int(activation_type),
                     n_group=n_group,
                     topk_group=topk_group,
                     norm_topk_prob=norm_topk_prob,
                     routed_scaling_factor=routed_scaling_factor,
                     routing_mode=int(routing_mode),
                     max_routed_rows=_MAX_ROUTED_ROWS_AUTO)


def nvfp4_moe(router_logits: Tensor,
              hidden_states: Tensor,
              moe_weights: dict,
              num_experts: int,
              top_k: int,
              hidden_size: int,
              moe_inter_size: int,
              activation_type: MoeActivation,
              n_group: int,
              topk_group: int,
              norm_topk_prob: int,
              routed_scaling_factor: float,
              routing_mode: MoeRouting,
              sm12x: bool,
              *,
              weight_prefix: str,
              weight_bindings: "dict[str, dict] | None" = None) -> Tensor:
    """Run NVFP4 mixture-of-experts and return token-major ``[T,H]`` FP16.

    ``moe_weights`` provides the 9 constant inputs as numpy arrays:
    ``fc1_qweights, fc1_blocks_scale, fc1_alpha, fc2_qweights,
    fc2_blocks_scale, fc2_alpha, input_global_scale, down_input_scale,
    e_score_correction_bias``.

    Model families provide checkpoint bindings when their provider layout can
    be rebuilt at runtime. Padded or fused expert banks omit them and remain
    constants in checkpoint-direct builds.
    """
    _validate_token_layout(router_logits, hidden_states)

    order = [
        ("fc1_qweights", np.int8),
        ("fc1_blocks_scale", np.int8),
        ("fc1_alpha", np.float32),
        ("fc2_qweights", np.int8),
        ("fc2_blocks_scale", np.int8),
        ("fc2_alpha", np.float32),
        ("input_global_scale", np.float32),
        ("down_input_scale", np.float32),
        ("e_score_correction_bias", np.float32),
    ]
    bindings = weight_bindings or {}
    weight_inputs = [
        parameter(weight_prefix + "." + name,
                  _parameter_value(moe_weights[name], dt),
                  "nvfp4_moe",
                  recipe=bindings.get(name)) for name, dt in order
    ]
    inputs = [router_logits, hidden_states] + weight_inputs
    operation_name = "nvfp4_moe_sm12x" if sm12x else "nvfp4_moe"
    return operation(operation_name,
                     inputs,
                     num_experts=num_experts,
                     top_k=top_k,
                     hidden_size=hidden_size,
                     moe_inter_size=moe_inter_size,
                     activation_type=int(activation_type),
                     n_group=n_group,
                     topk_group=topk_group,
                     norm_topk_prob=norm_topk_prob,
                     routed_scaling_factor=routed_scaling_factor,
                     routing_mode=int(routing_mode),
                     backend=_BACKEND_AUTO,
                     max_routed_rows=_MAX_ROUTED_ROWS_AUTO,
                     io_dtype=_IO_DTYPE_FP16)
