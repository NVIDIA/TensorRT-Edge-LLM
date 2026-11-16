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
"""Symbolic TensorRT tensor facade used by model definitions.

The builder still owns raw ``trt.ITensor`` values at network boundaries. This
wrapper is only for model code readability; backend lowering is resolved from
the active build scope so common symbolic expressions look like PyTorch:

``x = x.reshape((0, 0, hidden)); z = x + y; hidden = gate.silu() * up``.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import tensorrt as trt

from .scope import current_net

Scalar = Union[int, float, np.ndarray]

_TRT_TO_NUMPY = {
    trt.float16: np.float16,
    trt.float32: np.float32,
    trt.int8: np.int8,
    trt.int32: np.int32,
    trt.int64: np.int64,
    trt.uint8: np.uint8,
    trt.bool: np.bool_,
}


class Tensor:
    """Symbolic tensor used by model definitions and operations."""

    __array_priority__ = 10000

    def __init__(self, value: "trt.ITensor") -> None:
        self._value = value

    @staticmethod
    def unwrap(value):
        return value._value if isinstance(value, Tensor) else value

    def _as_trt(self) -> "trt.ITensor":
        """Return the backend tensor at an internal lowering boundary."""
        return self._value

    @property
    def shape(self):
        return self._value.shape

    @property
    def dtype(self):
        return self._value.dtype

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def ndim(self) -> int:
        return self.rank

    @property
    def name(self):
        return self._value.name

    @name.setter
    def name(self, value: str) -> None:
        self._value.name = value

    def _wrap(self, value) -> "Tensor":
        return Tensor(value)

    def _constant_like(self, value: Scalar, name: str = "scalar"):
        try:
            dtype = _TRT_TO_NUMPY[self.dtype]
        except KeyError as error:
            raise TypeError(
                f"cannot create a constant matching TensorRT dtype {self.dtype}"
            ) from error
        arr = np.asarray(value, dtype=dtype)
        if arr.ndim == 0:
            arr = arr.reshape((1, ) * len(self.shape))
        return current_net().const(arr, name)

    def _binary(self, other, op: "trt.ElementWiseOperation") -> "Tensor":
        rhs = Tensor.unwrap(other)
        if not isinstance(rhs, trt.ITensor):
            rhs = self._constant_like(rhs)
        return self._wrap(current_net().elementwise(self._value, rhs, op))

    def __add__(self, other) -> "Tensor":
        return self._binary(other, trt.ElementWiseOperation.SUM)

    def __radd__(self, other) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other) -> "Tensor":
        return self._binary(other, trt.ElementWiseOperation.SUB)

    def __rsub__(self, other) -> "Tensor":
        lhs = self._constant_like(other)
        return self._wrap(current_net().elementwise(
            lhs, self._value, trt.ElementWiseOperation.SUB))

    def __mul__(self, other) -> "Tensor":
        return self._binary(other, trt.ElementWiseOperation.PROD)

    def __rmul__(self, other) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Tensor":
        return self._binary(other, trt.ElementWiseOperation.DIV)

    def __rtruediv__(self, other) -> "Tensor":
        lhs = self._constant_like(other)
        return self._wrap(current_net().elementwise(
            lhs, self._value, trt.ElementWiseOperation.DIV))

    def __neg__(self) -> "Tensor":
        return self._wrap(current_net().unary(self._value,
                                              trt.UnaryOperation.NEG))

    def __getitem__(self, index) -> "Tensor":
        """Apply basic PyTorch-style integer and unit-stride slicing."""
        indices = index if isinstance(index, tuple) else (index, )
        if sum(item is Ellipsis for item in indices) > 1:
            raise IndexError("an index can contain only one ellipsis")
        if any(item is None for item in indices):
            raise IndexError("use unsqueeze() instead of None indexing")

        rank = len(self.shape)
        specified = sum(item is not Ellipsis for item in indices)
        if specified > rank:
            raise IndexError(f"too many indices for a rank-{rank} tensor")
        expanded = []
        for item in indices:
            if item is Ellipsis:
                expanded.extend([slice(None)] * (rank - specified))
            else:
                expanded.append(item)
        expanded.extend([slice(None)] * (rank - len(expanded)))

        result = self
        axis = 0
        for item in expanded:
            if isinstance(item, int):
                dimension = int(result.shape[axis])
                position = item
                if position < 0:
                    if dimension < 0:
                        raise IndexError(
                            "negative indexing requires a static dimension")
                    position += dimension
                if position < 0 or (dimension >= 0 and position >= dimension):
                    raise IndexError(f"index {item} is out of bounds")
                gathered = current_net().gather(
                    result._value, np.asarray(position, dtype=np.int64), axis)
                result = result._wrap(current_net().squeeze(
                    gathered, axis, len(result.shape)))
                continue
            if not isinstance(item, slice):
                raise TypeError(
                    f"Tensor indices must be integers or slices, got {type(item).__name__}"
                )
            if item.step not in (None, 1):
                raise IndexError(
                    "only unit-stride Tensor slices are supported")
            if item.start is None and item.stop is None:
                axis += 1
                continue

            dimension = int(result.shape[axis])
            start = 0 if item.start is None else int(item.start)
            stop = dimension if item.stop is None else int(item.stop)
            if start < 0 or stop < 0:
                if dimension < 0:
                    raise IndexError(
                        "negative slicing requires a static dimension")
                start = start + dimension if start < 0 else start
                stop = stop + dimension if stop < 0 else stop
            if stop < 0:
                raise IndexError("slice stop requires a static dimension")
            if dimension >= 0:
                start = min(max(start, 0), dimension)
                stop = min(max(stop, start), dimension)
            result = result._wrap(current_net().slice_axis(
                result._value, axis, start, max(0, stop - start),
                len(result.shape)))
            axis += 1
        return result

    def maximum(self, other) -> "Tensor":
        return self._binary(other, trt.ElementWiseOperation.MAX)

    def minimum(self, other) -> "Tensor":
        return self._binary(other, trt.ElementWiseOperation.MIN)

    def equal(self, other) -> "Tensor":
        return self._binary(other, trt.ElementWiseOperation.EQUAL)

    def reshape(self, shape: Sequence[int]) -> "Tensor":
        return self._wrap(current_net().reshape(self._value, shape))

    def transpose(self, permutation: Sequence[int]) -> "Tensor":
        return self._wrap(current_net().transpose(self._value, permutation))

    def cast(self, dtype: "trt.DataType") -> "Tensor":
        return self._wrap(current_net().cast(self._value, dtype))

    def to(self, dtype: "trt.DataType") -> "Tensor":
        return self.cast(dtype)

    def unsqueeze(self, axis: int, rank: int) -> "Tensor":
        return self._wrap(current_net().unsqueeze(self._value, axis, rank))

    def squeeze(self, axis: int, rank: int) -> "Tensor":
        return self._wrap(current_net().squeeze(self._value, axis, rank))

    def slice_last_dim(self, offset: int, size: int, rank: int) -> "Tensor":
        return self._wrap(current_net().slice_last_dim(self._value, offset,
                                                       size, rank))

    def slice_axis(self, axis: int, offset: int, size: int,
                   rank: int) -> "Tensor":
        return self._wrap(current_net().slice_axis(self._value, axis, offset,
                                                   size, rank))

    def silu(self) -> "Tensor":
        return self._wrap(current_net().silu(self._value))

    def relu(self) -> "Tensor":
        return self._wrap(current_net().relu(self._value))

    def tanh(self) -> "Tensor":
        return self._wrap(current_net().tanh(self._value))

    def sigmoid(self) -> "Tensor":
        return self._wrap(current_net().activation(self._value,
                                                   trt.ActivationType.SIGMOID))

    def sin(self) -> "Tensor":
        return self._wrap(current_net().unary(self._value,
                                              trt.UnaryOperation.SIN))

    def cos(self) -> "Tensor":
        return self._wrap(current_net().unary(self._value,
                                              trt.UnaryOperation.COS))

    def sqrt(self) -> "Tensor":
        return self._wrap(current_net().unary(self._value,
                                              trt.UnaryOperation.SQRT))

    def abs(self) -> "Tensor":
        return self._wrap(current_net().unary(self._value,
                                              trt.UnaryOperation.ABS))

    def sign(self) -> "Tensor":
        return self._wrap(current_net().unary(self._value,
                                              trt.UnaryOperation.SIGN))

    def log(self) -> "Tensor":
        return self._wrap(current_net().unary(self._value,
                                              trt.UnaryOperation.LOG))

    def exp(self) -> "Tensor":
        return self._wrap(current_net().unary(self._value,
                                              trt.UnaryOperation.EXP))

    def elu(self) -> "Tensor":
        return self._wrap(current_net().activation(self._value,
                                                   trt.ActivationType.ELU))

    def gelu(self) -> "Tensor":
        return self._wrap(current_net().gelu(self._value))

    def gelu_tanh(self) -> "Tensor":
        return self._wrap(current_net().gelu_tanh(self._value))

    def activation(self, name: str) -> "Tensor":
        """Apply a Transformers ``ACT2FN`` activation by configuration name."""
        if name in ("silu", "swish"):
            return self.silu()
        if name == "sigmoid":
            return self.sigmoid()
        if name == "gelu":
            return self.gelu()
        if name in ("gelu_new", "gelu_pytorch_tanh"):
            return self.gelu_tanh()
        if name == "quick_gelu":
            return self * (self * 1.702).sigmoid()
        if name == "relu":
            return self.relu()
        if name == "relu2":
            activated = self.relu()
            return activated * activated
        raise ValueError(f"unsupported activation {name!r}")

    def softmax(self, axis: int) -> "Tensor":
        return self._wrap(current_net().softmax(self._value, axis))

    def log_softmax(self, axis: int) -> "Tensor":
        return self._wrap(current_net().log_softmax(self._value, axis))

    def gather(self,
               indices,
               axis: int = 0,
               num_elementwise_dims: int = 0) -> "Tensor":
        return self._wrap(current_net().gather_tensor(self._value,
                                                      Tensor.unwrap(indices),
                                                      axis,
                                                      num_elementwise_dims))

    def reduce(self, op: "trt.ReduceOperation", axes: int,
               keep_dims: bool) -> "Tensor":
        return self._wrap(current_net().reduce(self._value, op, axes,
                                               keep_dims))

    def _reduce_dims(self, dimensions) -> int:
        dimensions = ((dimensions, )
                      if isinstance(dimensions, int) else tuple(dimensions))
        axes = 0
        for dimension in dimensions:
            normalized = dimension if dimension >= 0 else dimension + self.rank
            if not 0 <= normalized < self.rank:
                raise IndexError(
                    f"dimension {dimension} is invalid for rank {self.rank}")
            axes |= 1 << normalized
        return axes

    def sum(self, dim, keepdim: bool = False) -> "Tensor":
        """Reduce dimensions using PyTorch-style ``Tensor.sum`` syntax."""
        return self.reduce(trt.ReduceOperation.SUM, self._reduce_dims(dim),
                           keepdim)

    def mean(self, dim, keepdim: bool = False) -> "Tensor":
        """Reduce dimensions using PyTorch-style ``Tensor.mean`` syntax."""
        return self.reduce(trt.ReduceOperation.AVG, self._reduce_dims(dim),
                           keepdim)

    def matmul(
            self,
            other,
            lhs_op: "trt.MatrixOperation" = trt.MatrixOperation.NONE,
            rhs_op: "trt.MatrixOperation" = trt.MatrixOperation.NONE
    ) -> "Tensor":
        return self._wrap(current_net().matmul(self._value, other, lhs_op,
                                               rhs_op))

    def __matmul__(self, other) -> "Tensor":
        return self.matmul(other)
