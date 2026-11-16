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
"""Closed, shape-checked tensor transforms for checkpoint conversion."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

from .schema import ShapeMismatch

Shape = tuple[int, ...]


class TransformKind(str, enum.Enum):
    """Supported source tensor layout operations."""

    IDENTITY = "identity"
    TRANSPOSE = "transpose"
    RESHAPE = "reshape"
    TRANSPOSE_RESHAPE = "transpose_reshape"
    SPLIT_TRANSPOSE_RESHAPE = "split_transpose_reshape"


@dataclass(frozen=True)
class Transform:
    """One declarative layout transform.

    Attributes:
        kind: Closed transform operation.
        axes: Axis order applied before optional reshaping.
        output_shapes: Exact shapes produced by the transform.
        split_axis: Axis split into one output per slice.
    """

    kind: TransformKind
    axes: tuple[int, ...] = ()
    output_shapes: tuple[Shape, ...] = ()
    split_axis: int = 0


@dataclass(frozen=True)
class MappingRule:
    """Map one source leaf to one or more target tensors.

    Attributes:
        source_path: Tuple path in the canonical restored Flax tree.
        source_shape: Exact accepted source shape.
        target_names: Exact HF checkpoint tensor names.
        target_shapes: Expected shapes corresponding to ``target_names``.
        transform: Layout operation applied during materialization.
    """

    source_path: tuple[str, ...]
    source_shape: Shape
    target_names: tuple[str, ...]
    target_shapes: tuple[Shape, ...]
    transform: Transform

    @property
    def target_bytes(self) -> int:
        """Return uncompressed BF16 payload bytes for all rule targets."""
        elements = sum(_numel(shape) for shape in self.target_shapes)
        return elements * 2


@dataclass(frozen=True)
class MappingPlan:
    """Immutable, complete source-to-target mapping plan.

    Attributes:
        rules: Deterministically ordered mapping rules.
        variant: Exact source variant name.
    """

    rules: tuple[MappingRule, ...]
    variant: str

    @property
    def target_names(self) -> tuple[str, ...]:
        """Return all target names in rule order."""
        return tuple(name for rule in self.rules for name in rule.target_names)


def _numel(shape: Shape) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements


def _validate_shape(shape: Shape, label: str) -> None:
    if not isinstance(shape, tuple):
        raise ShapeMismatch(
            f"{label} must be a tuple, found {type(shape).__name__}")
    if any(
            isinstance(dimension, bool) or not isinstance(dimension, int)
            or dimension <= 0 for dimension in shape):
        raise ShapeMismatch(f"{label} contains an invalid dimension: {shape}")


def _shape(tensor: Any) -> Shape:
    return tuple(int(dimension) for dimension in tensor.shape)


def _transpose(tensor: Any, axes: tuple[int, ...]) -> Any:
    if not axes:
        return tensor
    return tensor.transpose(axes)


def _validate_transform(source_shape: Shape, transform: Transform) -> None:
    _validate_shape(source_shape, "source shape")
    if not isinstance(transform.kind, TransformKind):
        raise ValueError(f"unsupported transform: {transform.kind}")
    if not isinstance(transform.axes, tuple):
        raise ShapeMismatch("transform axes must be a tuple")
    if not isinstance(transform.output_shapes,
                      tuple) or not transform.output_shapes:
        raise ShapeMismatch("transform must declare at least one output shape")
    for index, output_shape in enumerate(transform.output_shapes):
        _validate_shape(output_shape, f"transform output shape {index}")

    single_output_kinds = {
        TransformKind.IDENTITY,
        TransformKind.TRANSPOSE,
        TransformKind.RESHAPE,
        TransformKind.TRANSPOSE_RESHAPE,
    }
    if transform.kind in single_output_kinds and len(
            transform.output_shapes) != 1:
        raise ShapeMismatch(
            f"{transform.kind.value} requires exactly one output shape")

    source_rank = len(source_shape)
    if transform.kind in (TransformKind.TRANSPOSE,
                          TransformKind.TRANSPOSE_RESHAPE):
        if (transform.axes and tuple(sorted(transform.axes)) != tuple(
                range(source_rank))):
            raise ShapeMismatch(
                f"transpose axes {transform.axes} are not a permutation of "
                f"rank {source_rank}")
    elif transform.kind == TransformKind.SPLIT_TRANSPOSE_RESHAPE:
        if (isinstance(transform.split_axis, bool)
                or not isinstance(transform.split_axis, int)
                or not 0 <= transform.split_axis < source_rank):
            raise ShapeMismatch(
                f"invalid split axis {transform.split_axis} for rank {source_rank}"
            )
        split_count = source_shape[transform.split_axis]
        if split_count != len(transform.output_shapes):
            raise ShapeMismatch(
                f"split produced {split_count} outputs, expected "
                f"{len(transform.output_shapes)}")
        part_rank = source_rank - 1
        if (transform.axes
                and tuple(sorted(transform.axes)) != tuple(range(part_rank))):
            raise ShapeMismatch(
                f"split transpose axes {transform.axes} are not a permutation "
                f"of rank {part_rank}")
        part_elements = _numel(source_shape) // split_count
        if any(
                _numel(output_shape) != part_elements
                for output_shape in transform.output_shapes):
            raise ShapeMismatch(
                "split transform output shape changes the per-slice element count"
            )
    elif transform.axes:
        raise ShapeMismatch(
            f"{transform.kind.value} does not accept transpose axes")

    if transform.kind in (TransformKind.RESHAPE,
                          TransformKind.TRANSPOSE_RESHAPE):
        if _numel(transform.output_shapes[0]) != _numel(source_shape):
            raise ShapeMismatch(
                "transform output shape changes the source element count")


def apply_transform(source_tensor: Any,
                    transform: Transform) -> tuple[Any, ...]:
    """Apply a transform without materializing a host Torch tensor.

    Args:
        source_tensor: JAX-compatible tensor object.
        transform: Closed transform descriptor.

    Returns:
        Tuple of transformed tensors.

    Raises:
        ShapeMismatch: If output count or shapes do not match the descriptor.
        ValueError: If the transform kind is unsupported.
    """
    source_shape = _shape(source_tensor)
    _validate_transform(source_shape, transform)
    handlers = {
        TransformKind.IDENTITY: _apply_identity,
        TransformKind.TRANSPOSE: _apply_transpose,
        TransformKind.RESHAPE: _apply_reshape,
        TransformKind.TRANSPOSE_RESHAPE: _apply_transpose_reshape,
        TransformKind.SPLIT_TRANSPOSE_RESHAPE: _apply_split_transpose_reshape,
    }
    try:
        outputs = handlers[transform.kind](source_tensor, transform)
    except KeyError as error:
        raise ValueError(f"unsupported transform: {transform.kind}") from error

    actual_shapes = tuple(_shape(output) for output in outputs)
    if actual_shapes != transform.output_shapes:
        raise ShapeMismatch(
            f"transform produced {actual_shapes}, expected {transform.output_shapes}"
        )
    return outputs


def _apply_identity(source_tensor: Any,
                    transform: Transform) -> tuple[Any, ...]:
    return (source_tensor, )


def _apply_transpose(source_tensor: Any,
                     transform: Transform) -> tuple[Any, ...]:
    return (_transpose(source_tensor, transform.axes), )


def _apply_reshape(source_tensor: Any,
                   transform: Transform) -> tuple[Any, ...]:
    return (source_tensor.reshape(transform.output_shapes[0]), )


def _apply_transpose_reshape(source_tensor: Any,
                             transform: Transform) -> tuple[Any, ...]:
    output = _transpose(source_tensor,
                        transform.axes).reshape(transform.output_shapes[0])
    return (output, )


def _apply_split_transpose_reshape(source_tensor: Any,
                                   transform: Transform) -> tuple[Any, ...]:
    outputs = []
    for index, output_shape in enumerate(transform.output_shapes):
        selectors = [slice(None)] * len(source_tensor.shape)
        selectors[transform.split_axis] = index
        part = _transpose(source_tensor[tuple(selectors)], transform.axes)
        outputs.append(part.reshape(output_shape))
    return tuple(outputs)


def _round_fp32_bits_to_bf16(host_fp32: Any) -> Any:
    import numpy as np

    bits = host_fp32.view(np.uint32)
    is_nan = ((bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)) & (
        (bits & np.uint32(0x007FFFFF)) != 0)
    bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = ((bits + bias) >> np.uint32(16)).astype(np.uint16)
    quiet_nan = ((bits >> np.uint32(16)) | np.uint32(0x0040)).astype(np.uint16)
    return np.ascontiguousarray(np.where(is_nan, quiet_nan, rounded))


def _bf16_tensor_from_bits(bf16_bits: Any) -> Any:
    import numpy as np
    import torch

    storage = np.ascontiguousarray(bf16_bits, dtype=np.uint16)
    return torch.from_numpy(storage.view(np.int16)).view(torch.bfloat16)


def convert_fp32_to_bf16(tensor: Any) -> Any:
    """Materialize canonical FP32 on CPU and round it to contiguous BF16.

    Args:
        tensor: Canonical JAX-compatible FP32 source tensor.

    Returns:
        Writable, contiguous CPU Torch BF16 tensor rounded to nearest even.

    Raises:
        ShapeMismatch: If the source is not FP32 or the target representation
            is not contiguous CPU BF16.
    """
    import numpy as np

    try:
        import jax
    except ImportError:
        device_value = tensor
    else:
        try:
            device_value = jax.device_get(tensor)
        except (TypeError, ValueError) as error:
            source_dtype = getattr(tensor, "dtype", "unknown")
            raise ShapeMismatch(
                f"source tensor must be FP32, found {source_dtype}") from error
    try:
        host = np.asarray(device_value)
    except (TypeError, ValueError) as error:
        source_dtype = getattr(device_value, "dtype",
                               getattr(tensor, "dtype", "unknown"))
        raise ShapeMismatch(
            f"source tensor must be FP32, found {source_dtype}") from error
    if host.dtype != np.dtype(np.float32):
        raise ShapeMismatch(f"source tensor must be FP32, found {host.dtype}")
    host_fp32 = np.array(host, dtype=np.float32, order="C", copy=True)
    if not host_fp32.flags.c_contiguous or not host_fp32.flags.writeable:
        raise ShapeMismatch(
            "materialized FP32 source must be contiguous and writable")
    target = _bf16_tensor_from_bits(_round_fp32_bits_to_bf16(host_fp32))
    import torch
    if (target.dtype != torch.bfloat16 or target.device.type != "cpu"
            or not target.is_contiguous()
            or tuple(target.shape) != tuple(host_fp32.shape)):
        raise ShapeMismatch("FP32-to-BF16 materialization is invalid")
    return target


def convert_tensor(source_tensor: Any, rule: MappingRule) -> dict[str, Any]:
    """Apply one rule and materialize exact host BF16 target tensors.

    Args:
        source_tensor: Source JAX tensor selected by ``rule.source_path``.
        rule: Validated mapping rule.

    Returns:
        Target-name-keyed contiguous CPU Torch tensors.

    Raises:
        ShapeMismatch: If source, result, or dtype differs from the rule.
    """
    source_shape = _shape(source_tensor)
    _validate_shape(rule.source_shape, "rule source shape")
    if source_shape != rule.source_shape:
        raise ShapeMismatch(
            f"{rule.source_path} has shape {source_shape}, expected {rule.source_shape}"
        )
    if not isinstance(rule.target_shapes, tuple):
        raise ShapeMismatch(
            f"{rule.source_path} target shapes must be a tuple")
    if rule.transform.output_shapes != rule.target_shapes:
        raise ShapeMismatch(
            f"{rule.source_path} transform declares "
            f"{rule.transform.output_shapes}, expected targets {rule.target_shapes}"
        )
    if not isinstance(rule.target_names,
                      tuple) or not rule.target_names or len(
                          rule.target_names) != len(rule.target_shapes):
        raise ShapeMismatch(
            f"{rule.source_path} target name/shape count differs")
    if len(rule.target_names) != len(set(rule.target_names)) or any(
            not isinstance(name, str) or not name
            for name in rule.target_names):
        raise ShapeMismatch(
            f"{rule.source_path} target names must be unique non-empty strings"
        )
    for index, target_shape in enumerate(rule.target_shapes):
        _validate_shape(target_shape, f"rule target shape {index}")
    outputs = apply_transform(source_tensor, rule.transform)
    if len(outputs) != len(rule.target_names):
        raise ShapeMismatch(
            f"{rule.source_path} produced {len(outputs)} outputs for "
            f"{len(rule.target_names)} targets")
    return {
        name: convert_fp32_to_bf16(output)
        for name, output in zip(rule.target_names, outputs)
    }
