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
"""Core functional symbolic operations.

Functions infer their backend from the active :class:`Module` scope. Model
definitions therefore pass tensors and operation attributes, never a
TensorRT network object.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import tensorrt as trt

from ...core import quantization
from ..scope import current_context, current_net
from ..tensor import Tensor

Dimension = Union[int, Tensor]


def tensor(value) -> Tensor:
    """Return ``value`` as a symbolic tensor."""
    return value if isinstance(value, Tensor) else Tensor(value)


def unwrap(value):
    """Unwrap a symbolic tensor at an internal backend boundary."""
    return Tensor.unwrap(value)


def constant(value: object, name: str = "const") -> Tensor:
    """Create a typed TensorRT constant in the active graph."""
    return tensor(current_net().const(np.asarray(value), name))


def linear(input: Tensor, prefix: str, rank: int = 3):
    """Apply a checkpoint-backed linear projection."""
    context = current_context()
    if context.backend == "eager":
        return _linear_eager(context.weights, input, prefix)
    return tensor(_linear(context.net, context.weights, input, prefix, rank))


def normalization(input: Tensor, prefix: str, eps: float, rank: int = 3):
    """Apply checkpoint-backed LayerNorm or RMSNorm."""
    context = current_context()
    if context.backend == "eager":
        return _normalization_eager(context.weights, input, prefix, eps)
    return tensor(
        _normalization(context.net, context.weights, input, prefix, eps, rank))


def apply_rope(query: Tensor, key: Tensor, rotary: Tensor, num_heads: int,
               head_size: int) -> Tuple[Tensor, Tensor]:
    """Apply one-dimensional rotary embeddings."""
    query, key = _apply_rope(current_net(), query, key, rotary, num_heads,
                             head_size)
    return tensor(query), tensor(key)


def apply_multidimensional_rope(
        query: Tensor,
        key: Tensor,
        rotary: Tensor,
        head_size: int,
        num_dimensions: int = 2) -> Tuple[Tensor, Tensor]:
    """Apply independent rotary channels for spatial dimensions."""
    query, key = _apply_multidimensional_rope(current_net(), query, key,
                                              rotary, head_size,
                                              num_dimensions)
    return tensor(query), tensor(key)


def batch_token(input: Tensor, token: np.ndarray) -> Tensor:
    return tensor(_batch_token(current_net(), input, token))


def pixel_unshuffle(input: Tensor, side: int, scale: int,
                    hidden_size: int) -> Tensor:
    return tensor(
        _pixel_unshuffle(current_net(), input, side, scale, hidden_size))


def embedding_lookup(table, indices: Tensor, axis: int = 0) -> Tensor:
    """Gather rows from a constant or checkpoint-backed embedding table."""
    if isinstance(table, Tensor):
        return table.gather(indices, axis)
    return tensor(_embedding_lookup(current_net(), table, indices, axis))


def fourier_features(values: Tensor, feature_count: int,
                     max_frequency: float) -> Tensor:
    return tensor(
        _fourier_features(current_net(), values, feature_count, max_frequency))


def linear_with_weights(input: Tensor,
                        weight: np.ndarray,
                        bias: Optional[np.ndarray] = None,
                        rank: int = 3) -> Tensor:
    return tensor(current_net().linear(input, weight, bias, rank))


def linear_from_weights(input: Tensor,
                        weights,
                        rank: int = 3,
                        name: str = "") -> Tensor:
    return tensor(current_net().linear_from_weights(input,
                                                    weights,
                                                    rank,
                                                    name=name))


def fused_nvfp4_gemm_all_reduce(input: Tensor,
                                weights,
                                bias,
                                tp_size: int,
                                rank: int = 3,
                                name: str = "",
                                bias_recipe=None) -> Tensor:
    return tensor(current_net().fused_nvfp4_gemm_all_reduce(
        input,
        weights,
        bias,
        tp_size,
        rank,
        name=name,
        bias_recipe=bias_recipe))


def linear_f32(input: Tensor,
               weight: np.ndarray,
               bias: Optional[np.ndarray] = None,
               rank: int = 3) -> Tensor:
    return tensor(current_net().linear_f32(input, weight, bias, rank))


def linear_f32_from_weights(input: Tensor,
                            weights,
                            name: str,
                            rank: int = 3) -> Tensor:
    return tensor(current_net().linear_f32_from_weights(
        input, weights, name, rank))


def dynamic_lora(input: Tensor, output: Tensor, prefix: str, in_features: int,
                 out_features: int) -> Tensor:
    return tensor(current_net().dynamic_lora(input, output, prefix,
                                             in_features, out_features))


def rms_norm(input: Tensor,
             weight,
             eps: float,
             rank: int = 3,
             weight_before_cast: bool = False) -> Tensor:
    return tensor(current_net().rmsnorm(input,
                                        weight,
                                        eps,
                                        rank,
                                        weight_before_cast=weight_before_cast))


def layer_norm(input: Tensor, weight: np.ndarray, bias: np.ndarray, eps: float,
               rank: int) -> Tensor:
    return tensor(current_net().layernorm(input, weight, bias, eps, rank))


def convolution(input: Tensor,
                weight,
                bias=None,
                stride: Sequence[int] = (1, ),
                padding: Sequence[int] = (0, ),
                dilation: Sequence[int] = (1, ),
                groups: int = 1,
                pre_padding: Optional[Sequence[int]] = None,
                post_padding: Optional[Sequence[int]] = None) -> Tensor:
    return tensor(current_net().convolution(input, weight, bias, stride,
                                            padding, dilation, groups,
                                            pre_padding, post_padding))


def deconvolution(input: Tensor,
                  weight: np.ndarray,
                  bias: Optional[np.ndarray] = None,
                  stride: Sequence[int] = (1, ),
                  padding: Sequence[int] = (0, ),
                  groups: int = 1,
                  pre_padding: Optional[Sequence[int]] = None,
                  post_padding: Optional[Sequence[int]] = None) -> Tensor:
    return tensor(current_net().deconvolution(input, weight, bias, stride,
                                              padding, groups, pre_padding,
                                              post_padding))


def concatenate(inputs: Sequence[Tensor], dim: int = 0) -> Tensor:
    return tensor(current_net().concat(inputs, dim))


def cast(input: Tensor, dtype: "trt.DataType") -> Tensor:
    return tensor(input).cast(dtype)


def reshape(input: Tensor, shape: Sequence[int]) -> Tensor:
    return tensor(input).reshape(shape)


def dynamic_reshape(input: Tensor,
                    shape: Union[Tensor, Sequence[Dimension]]) -> Tensor:
    """Reshape using a runtime shape vector assembled from scalar tensors."""
    return tensor(current_net().dynamic_reshape(
        input, _dimension_vector(shape, "reshape_dim")))


def slice_last_dim(input: Tensor, offset: int, size: int, rank: int) -> Tensor:
    return tensor(current_net().slice_last_dim(input, offset, size, rank))


def pad_last_dim(input: Tensor, padding: int, rank: int) -> Tensor:
    return tensor(current_net().pad_last_dim(input, padding, rank))


def empty_sequence(input: Tensor, last_dim: int) -> Tensor:
    return tensor(current_net().empty_sequence(input, last_dim))


def gather_last_tokens(input: Tensor, indices: Tensor) -> Tensor:
    return tensor(current_net().gather_last_tokens(input, indices))


def gather_nd(input: Tensor,
              indices: Tensor,
              num_elementwise_dims: int = 0) -> Tensor:
    return tensor(current_net().gather_nd(input, indices,
                                          num_elementwise_dims))


def topk(input: Tensor, k: int, axis: int) -> Tuple[Tensor, Tensor]:
    values, indices = current_net().topk(input, k, axis)
    return tensor(values), tensor(indices)


def reduce(input: Tensor,
           op: "trt.ReduceOperation",
           axes: int,
           keep_dims: bool = False) -> Tensor:
    return tensor(current_net().reduce(input, op, axes, keep_dims))


def select(condition: Tensor, when_true: Tensor, when_false: Tensor) -> Tensor:
    return tensor(current_net().select(condition, when_true, when_false))


def rotary_embedding(input: Tensor,
                     cos_cache: Tensor,
                     sin_cache: Tensor,
                     position_ids: Tensor,
                     rotary_dim: int,
                     interleaved: bool = False) -> Tensor:
    """Apply rotary embedding."""
    return tensor(current_net().rotary_embedding(input, cos_cache, sin_cache,
                                                 position_ids, rotary_dim,
                                                 interleaved))


def kv_cache_update(cache: Tensor, update: Tensor,
                    write_indices: Tensor) -> Tensor:
    """Update a linear TensorRT KV cache."""
    return tensor(current_net().kv_cache_update(cache, update, write_indices))


def scaled_dot_product_attention(query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 mask: Optional[Tensor] = None,
                                 key_value_lengths: Optional[Tensor] = None,
                                 scale: Optional[float] = None,
                                 is_causal: bool = False) -> Tensor:
    """Apply scaled dot-product attention."""
    return tensor(current_net().scaled_dot_product_attention(
        query, key, value, mask, key_value_lengths, scale, is_causal))


def matmul(lhs: Tensor,
           rhs: Tensor,
           transpose_lhs: bool = False,
           transpose_rhs: bool = False) -> Tensor:
    lhs_op = (trt.MatrixOperation.TRANSPOSE
              if transpose_lhs else trt.MatrixOperation.NONE)
    rhs_op = (trt.MatrixOperation.TRANSPOSE
              if transpose_rhs else trt.MatrixOperation.NONE)
    return tensor(lhs).matmul(rhs, lhs_op=lhs_op, rhs_op=rhs_op)


def shape_of(input: Tensor) -> Tensor:
    return tensor(current_net().shape_of(input))


def dynamic_slice(input: Tensor,
                  start: Union[Tensor, Sequence[Dimension]],
                  size: Union[Tensor, Sequence[Dimension]],
                  stride: Optional[Sequence[int]] = None) -> Tensor:
    """Slice with runtime start and size dimensions."""
    rank = tensor(input).rank
    strides = tuple(stride or [1] * rank)
    return tensor(current_net().dynamic_slice(
        input, _dimension_vector(start, "slice_start"),
        _dimension_vector(size, "slice_size"), strides))


def _dimension_vector(values: Union[Tensor, Sequence[Dimension]],
                      name: str) -> Tensor:
    if isinstance(values, Tensor):
        return values
    dimensions = []
    for value in values:
        if isinstance(value, Tensor):
            dimensions.append(value)
        else:
            dimensions.append(
                constant(np.asarray([value], dtype=np.int32), name))
    return concatenate(dimensions, 0)


def _linear_precision(weights, prefix: str) -> str:
    if weights.has(prefix + ".weight_scale_2"):
        return quantization.QUANT_NVFP4
    if weights.has(prefix + ".qweight"):
        return weights.quant.quant_type
    if weights.has(prefix + ".weight_scale"):
        dtype = weights.store.dtype(weights._resolve(prefix + ".weight"))
        scale_dtype = weights.store.dtype(
            weights._resolve(prefix + ".weight_scale"))
        if dtype.startswith("F8") and scale_dtype == "U8":
            return quantization.QUANT_MXFP8
        if dtype.startswith("F8"):
            scale_shape = weights.store.shape(
                weights._resolve(prefix + ".weight_scale"))
            if len(scale_shape) == 4:
                return quantization.QUANT_FP8_BLOCK
            return quantization.QUANT_FP8
        if dtype == "I8":
            return quantization.QUANT_INT8_SQ
        if dtype == "U8":
            return quantization.QUANT_INT4_AWQ_MODELOPT
    return quantization.QUANT_FP16


def _linear(net, weights, hidden, prefix: str, rank: int):
    descriptor = weights.linear_descriptor(prefix,
                                           _linear_precision(weights, prefix))
    return net.linear_from_weights(hidden, descriptor, rank, name=prefix)


def _linear_eager(weights, hidden, prefix: str):
    import torch

    precision = _linear_precision(weights, prefix)
    if precision != quantization.QUANT_FP16:
        raise NotImplementedError(
            f"eager linear currently supports fp16 weights, got {precision}")
    descriptor = weights.linear(prefix, precision)
    weight = torch.as_tensor(descriptor.weight,
                             device=hidden.device,
                             dtype=hidden.dtype)
    bias = (None if descriptor.bias is None else torch.as_tensor(
        descriptor.bias, device=hidden.device, dtype=hidden.dtype))
    return torch.nn.functional.linear(hidden, weight, bias)


def _normalization(net, weights, hidden, prefix: str, eps: float, rank: int):
    weight = weights.fp16_parameter(prefix + ".weight")
    if weights.has(prefix + ".bias"):
        return net.layernorm(hidden, weight,
                             weights.fp16_parameter(prefix + ".bias"), eps,
                             rank)
    return net.rmsnorm(hidden, weight, eps, rank)


def _normalization_eager(weights, hidden, prefix: str, eps: float):
    import torch

    weight = torch.as_tensor(weights.f16(prefix + ".weight"),
                             device=hidden.device,
                             dtype=hidden.dtype)
    if weights.has(prefix + ".bias"):
        bias = torch.as_tensor(weights.f16(prefix + ".bias"),
                               device=hidden.device,
                               dtype=hidden.dtype)
        return torch.nn.functional.layer_norm(hidden, weight.shape, weight,
                                              bias, eps)
    variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = hidden.float() * torch.rsqrt(variance + eps)
    return (normalized.to(hidden.dtype) * weight).to(hidden.dtype)


def _apply_rope(net, query, key, rotary, num_heads: int, head_size: int):
    rotary = net.concat((rotary, rotary), 1)
    cosine = net.unary(rotary, trt.UnaryOperation.COS)
    sine = net.unary(rotary, trt.UnaryOperation.SIN)
    cosine = net.reshape(cosine, (0, 1, head_size))
    sine = net.reshape(sine, (0, 1, head_size))
    query = net.cast(query, trt.float32)
    key = net.cast(key, trt.float32)

    def rotate(tensor):
        first = net.slice_last_dim(tensor, 0, head_size // 2, 3)
        second = net.slice_last_dim(tensor, head_size // 2, head_size // 2, 3)
        negative = net.unary(second, trt.UnaryOperation.NEG)
        return net.concat((negative, first), 2)

    query = net.elementwise(
        net.elementwise(query, cosine, trt.ElementWiseOperation.PROD),
        net.elementwise(rotate(query), sine, trt.ElementWiseOperation.PROD),
        trt.ElementWiseOperation.SUM)
    key = net.elementwise(
        net.elementwise(key, cosine, trt.ElementWiseOperation.PROD),
        net.elementwise(rotate(key), sine, trt.ElementWiseOperation.PROD),
        trt.ElementWiseOperation.SUM)
    return net.cast(query, trt.float16), net.cast(key, trt.float16)


def _apply_multidimensional_rope(net, query, key, rotary, head_size: int,
                                 num_dimensions: int):
    """Apply independent rotary channels for each spatial dimension."""
    if head_size % (num_dimensions * 2):
        raise ValueError(f"head size {head_size} cannot be split across "
                         f"{num_dimensions} rotary dimensions")
    rotary = net.cast(rotary, trt.float32)
    cosine = net.unary(rotary, trt.UnaryOperation.COS)
    sine = net.unary(rotary, trt.UnaryOperation.SIN)
    width = head_size // num_dimensions

    def rotate(tensor):
        tensor = net.cast(tensor, trt.float32)
        pieces = []
        for index in range(num_dimensions):
            offset = index * width
            part = net.slice_last_dim(tensor, offset, width, 3)
            first = net.slice_last_dim(part, 0, width // 2, 3)
            second = net.slice_last_dim(part, width // 2, width // 2, 3)
            rotated = net.concat(
                (net.unary(second, trt.UnaryOperation.NEG), first), 2)
            cos_part = net.reshape(
                net.slice_last_dim(cosine, offset, width, 2), (0, 1, width))
            sin_part = net.reshape(net.slice_last_dim(sine, offset, width, 2),
                                   (0, 1, width))
            pieces.append(
                net.elementwise(
                    net.elementwise(part, cos_part,
                                    trt.ElementWiseOperation.PROD),
                    net.elementwise(rotated, sin_part,
                                    trt.ElementWiseOperation.PROD),
                    trt.ElementWiseOperation.SUM))
        return net.cast(net.concat(tuple(pieces), 2), trt.float16)

    return rotate(query), rotate(key)


def _batch_token(net, hidden, token: np.ndarray):
    carrier = net.slice_axis(hidden, 1, 0, 1, 3)
    zero = net.const(np.zeros((1, 1, 1), dtype=np.float16), "token_carrier")
    carrier = net.elementwise(carrier, zero, trt.ElementWiseOperation.PROD)
    token = np.asarray(token, dtype=np.float16)
    if token.ndim == 1:
        token = token.reshape(1, 1, -1)
    elif token.ndim == 2:
        token = token[None, :, :]
    elif token.ndim != 3:
        raise ValueError(
            f"class/register token must have rank 1-3: {token.shape}")
    constant = net.const(np.ascontiguousarray(token), "class_token")
    return net.elementwise(carrier, constant, trt.ElementWiseOperation.SUM)


def _pixel_unshuffle(net, hidden, side: int, scale: int, hidden_size: int):
    spatial = net.reshape(hidden, (0, side, side, hidden_size))
    spatial = net.reshape(
        spatial, (0, side // scale, scale, side // scale, scale, hidden_size))
    spatial = net.transpose(spatial, (0, 1, 3, 2, 4, 5))
    return net.reshape(spatial,
                       (0, (side // scale)**2, hidden_size * scale * scale))


def _embedding_lookup(net, table: np.ndarray, indices, axis: int = 0):
    table_tensor = net.const(table.astype(np.float16), "embedding")
    return net.gather_tensor(table_tensor, indices, axis)


def _fourier_features(net, values, feature_count: int, max_frequency: float):
    values = net.cast(values, trt.float32)
    half = feature_count // 2
    frequencies = np.logspace(0,
                              np.log10(max_frequency),
                              half,
                              dtype=np.float32)
    scalar_shape = (1, ) * len(values.shape)
    frequency_shape = (1, ) * (len(values.shape) - 1) + (half, )
    frequencies = net.const(frequencies.reshape(frequency_shape),
                            "frequencies")
    phase = net.elementwise(values, frequencies, trt.ElementWiseOperation.PROD)
    tau = net.const(
        np.array(2.0 * np.pi, dtype=np.float32).reshape(scalar_shape), "tau")
    phase = net.elementwise(phase, tau, trt.ElementWiseOperation.PROD)
    sine = net.unary(phase, trt.UnaryOperation.SIN)
    cosine = net.unary(phase, trt.UnaryOperation.COS)
    features = net.concat((sine, cosine), 2)
    scale = net.const(
        np.array(np.sqrt(2.0), dtype=np.float32).reshape(scalar_shape),
        "fourier_scale")
    features = net.elementwise(features, scale, trt.ElementWiseOperation.PROD)
    return net.cast(features, trt.float16)
