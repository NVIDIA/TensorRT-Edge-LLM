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
import tensorrt as trt

from ...core.weights import ParameterSpec
from ..scope import current_net
from ..tensor import Tensor
from ._operation import operation, parameter
from .core import topk

__all__ = [
    "MoeActivation",
    "MoeRouting",
    "fp16_moe",
    "int4_moe",
    "nvfp4_a16_moe",
    "nvfp4_moe",
    "trt_native_moe_fp16",
    "trt_native_moe_w4a8",
    "w2a16_moe",
]


def w2a16_moe(hidden_states: Tensor, buffers: dict, recipes: dict,
              topk_ids: Tensor, topk_weights: Tensor, num_experts: int,
              moe_inter_size: int, *, weight_prefix: str) -> Tensor:
    """Run packed IQ2_XXS experts with BF16 activations and FP32 routing.

    Buffers with a recipe are checkpoint-backed inputs when the ``iq2_xxs``
    weight kind is externalized; the rest are engine constants.
    """
    names = ("fc1_qweights", "fc2_qweights", "fc1_blocks_scale",
             "fc2_blocks_scale", "unit_scales", "lookup_table")
    inputs = [hidden_states]
    inputs.extend(
        parameter(f"{weight_prefix}.{name}",
                  buffers[name],
                  "iq2_xxs",
                  recipe=recipes.get(name)) for name in names)
    inputs.extend((topk_ids, topk_weights))
    return operation("w2a16_moe",
                     inputs,
                     num_experts=num_experts,
                     moe_inter_size=moe_inter_size)


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


def _expert_weight(name: str, value, shape: tuple, dtype) -> np.ndarray:
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.dtype(dtype):
        raise ValueError(
            f"{name} must be {np.dtype(dtype)} with shape {list(shape)}, got "
            f"{value.dtype} {list(value.shape)}")
    return value


def _trt_native_routing(router_logits: Tensor, top_k: int,
                        norm_topk_prob: int) -> tuple:
    """Softmax -> top-k (-> renormalise) routing, shaped ``[1, T, K]``."""
    if router_logits.dtype != trt.float32:
        raise ValueError(
            "router_logits must be float32 for the native MoE layer")
    scores, indices = topk(router_logits.softmax(1), top_k, 1)
    if norm_topk_prob:
        scores = scores / scores.sum(1, keepdim=True)
    return indices.reshape((1, -1, top_k)), scores.reshape((1, -1, top_k))


def trt_native_moe_fp16(router_logits: Tensor,
                        hidden_states: Tensor,
                        fc_gate_weights: np.ndarray,
                        fc_up_weights: np.ndarray,
                        fc_down_weights: np.ndarray,
                        num_experts: int,
                        top_k: int,
                        hidden_size: int,
                        moe_inter_size: int,
                        *,
                        norm_topk_prob: int = 1,
                        weight_prefix: str) -> Tensor:
    """Run FP16 SiLU-gated experts through TensorRT's native MoE layer.

    The layer takes precomputed routing, so softmax, top-k and the optional
    renormalisation are built here from ``router_logits`` (``[T, E]`` FP32).
    It is rank-3, so the token-major ``[T, H]`` block is wrapped as one
    sequence ``[1, T, H]`` / ``[1, T, K]`` and the output reshaped back.
    Weights are ``[E, H, I]`` (gate, up) and ``[E, I, H]`` (down) FP16 and
    stay engine constants; this revision has no externalized form for them.
    """
    _validate_token_layout(router_logits, hidden_states)
    gate = _expert_weight("fc_gate_weights", fc_gate_weights,
                          (num_experts, hidden_size, moe_inter_size),
                          np.float16)
    up = _expert_weight("fc_up_weights", fc_up_weights,
                        (num_experts, hidden_size, moe_inter_size), np.float16)
    down = _expert_weight("fc_down_weights", fc_down_weights,
                          (num_experts, moe_inter_size, hidden_size),
                          np.float16)
    indices, scores = _trt_native_routing(router_logits, top_k, norm_topk_prob)
    output = current_net().trt_native_moe(
        hidden_states.reshape((1, -1, hidden_size)),
        indices,
        scores,
        gate,
        up,
        down,
        name=weight_prefix + ".trt_native_moe")
    return Tensor(output).reshape((-1, hidden_size))


def trt_native_moe_w4a8(router_logits: Tensor,
                        hidden_states: Tensor,
                        moe_weights: dict,
                        num_experts: int,
                        top_k: int,
                        hidden_size: int,
                        moe_inter_size: int,
                        *,
                        norm_topk_prob: int = 1,
                        weight_prefix: str) -> Tensor:
    """Run NVFP4 SiLU-gated experts as W4A8 through TensorRT's native MoE layer.

    ``moe_weights`` carries the checkpoint's FP4 nibbles untouched:
    ``fc_gate_packed`` / ``fc_up_packed`` ``[E, I, H // 2]`` and
    ``fc_down_packed`` ``[E, H, I // 2]`` (uint8), their fp32 per-16 block
    scales ``fc_*_block_scale`` ``[E, out, in // 16]`` (already multiplied by
    the expert's global scale), the shared FP16 ``fc1_activation_scale``
    (static FP8 Q/DQ on the hidden states) and the per-expert fp32
    ``fc_down_activation_scales`` ``[E]``. Routing and rank handling follow
    :func:`trt_native_moe_fp16`. All buffers stay engine constants; this
    revision has no externalized form for them.
    """
    _validate_token_layout(router_logits, hidden_states)
    if hidden_size % 16 or moe_inter_size % 16:
        raise ValueError("W4A8 MoE needs hidden and intermediate sizes that "
                         "are multiples of the FP4 block (16)")

    def pair(projection: str, out_features: int, in_features: int) -> tuple:
        packed = _expert_weight(f"fc_{projection}_packed",
                                moe_weights[f"fc_{projection}_packed"],
                                (num_experts, out_features, in_features // 2),
                                np.uint8)
        scale = _expert_weight(f"fc_{projection}_block_scale",
                               moe_weights[f"fc_{projection}_block_scale"],
                               (num_experts, out_features, in_features // 16),
                               np.float32)
        return packed, scale

    gate = pair("gate", moe_inter_size, hidden_size)
    up = pair("up", moe_inter_size, hidden_size)
    down = pair("down", hidden_size, moe_inter_size)
    fc1_scale = _expert_weight("fc1_activation_scale",
                               moe_weights["fc1_activation_scale"], (),
                               np.float16)
    down_scales = _expert_weight("fc_down_activation_scales",
                                 moe_weights["fc_down_activation_scales"],
                                 (num_experts, ), np.float32)
    indices, scores = _trt_native_routing(router_logits, top_k, norm_topk_prob)
    output = current_net().trt_native_moe(
        hidden_states.reshape((1, -1, hidden_size)),
        indices,
        scores,
        gate,
        up,
        down,
        fc1_activation_scale=fc1_scale,
        fc_down_activation_scales=down_scales,
        name=weight_prefix + ".trt_native_moe")
    return Tensor(output).reshape((-1, hidden_size))


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
    bindings = weight_bindings
    inputs = [
        router_logits,
        hidden_states,
        parameter(weight_prefix + ".fc1_weights",
                  _parameter_value(weights["fc1_weights"], np.float16),
                  "fp16",
                  recipe=bindings.get("fc1_weights")),
        parameter(weight_prefix + ".fc2_weights",
                  _parameter_value(weights["fc2_weights"], np.float16),
                  "fp16",
                  recipe=bindings.get("fc2_weights")),
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
             zero_point_offset: int = 1,
             activation_type: MoeActivation = MoeActivation.SWIGLU) -> Tensor:
    """Run GPTQ-Marlin mixture-of-experts."""
    _validate_token_layout(router_logits, hidden_states)

    inputs = [
        router_logits,
        hidden_states,
        parameter(weight_prefix + ".fc_gate_up_qweights",
                  _parameter_value(weights["fc_gate_up_qweights"], np.int8),
                  "int4_moe",
                  recipe=weight_bindings.get("fc_gate_up_qweights")),
        parameter(weight_prefix + ".fc_gate_up_scales",
                  _parameter_value(weights["fc_gate_up_scales"], np.float16),
                  "int4_moe",
                  recipe=weight_bindings.get("fc_gate_up_scales")),
        parameter(weight_prefix + ".fc_down_qweights",
                  _parameter_value(weights["fc_down_qweights"], np.int8),
                  "int4_moe",
                  recipe=weight_bindings.get("fc_down_qweights")),
        parameter(weight_prefix + ".fc_down_scales",
                  _parameter_value(weights["fc_down_scales"], np.float16),
                  "int4_moe",
                  recipe=weight_bindings.get("fc_down_scales")),
    ]
    plugin_activation = {
        MoeActivation.SWIGLU: 0,
        MoeActivation.GEGLU: int(MoeActivation.GEGLU),
    }.get(activation_type)
    if plugin_activation is None:
        raise ValueError(
            f"INT4 MoE does not support activation {activation_type!r}")
    return operation("int4_moe",
                     inputs,
                     num_experts=num_experts,
                     top_k=top_k,
                     hidden_size=hidden_size,
                     moe_inter_size=moe_inter_size,
                     activation_type=plugin_activation,
                     quantization_group_size=group_size)


def nvfp4_a16_moe(router_logits: Tensor,
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
                  *,
                  weight_prefix: str,
                  weight_bindings: "dict[str, dict] | None" = None) -> Tensor:
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
                  _parameter_value(moe_weights[name], dtype),
                  "nvfp4_a16",
                  recipe=(None if weight_bindings is None else
                          weight_bindings[name])) for name, dtype in order
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
