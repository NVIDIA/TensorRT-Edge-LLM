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
"""Behavioral contracts for family-agnostic tensor transforms."""

import numpy as np
import pytest
import torch

from experimental.checkpoint_converters.schema import ShapeMismatch
from experimental.checkpoint_converters.transforms import (
    MappingRule, Transform, TransformKind, apply_transform,
    convert_fp32_to_bf16, convert_tensor)


@pytest.mark.parametrize(
    ("source", "transform", "expected"),
    [
        (
            np.arange(6).reshape(2, 3),
            Transform(TransformKind.IDENTITY, output_shapes=((2, 3), )),
            (np.arange(6).reshape(2, 3), ),
        ),
        (
            np.arange(6).reshape(2, 3),
            Transform(TransformKind.TRANSPOSE,
                      axes=(1, 0),
                      output_shapes=((3, 2), )),
            (np.arange(6).reshape(2, 3).T, ),
        ),
        (
            np.arange(6),
            Transform(TransformKind.RESHAPE, output_shapes=((2, 3), )),
            (np.arange(6).reshape(2, 3), ),
        ),
        (
            np.arange(24).reshape(2, 3, 4),
            Transform(TransformKind.TRANSPOSE_RESHAPE,
                      axes=(1, 0, 2),
                      output_shapes=((6, 4), )),
            (np.arange(24).reshape(2, 3, 4).transpose(1, 0, 2).reshape(6,
                                                                       4), ),
        ),
        (
            np.arange(12).reshape(2, 2, 3),
            Transform(
                TransformKind.SPLIT_TRANSPOSE_RESHAPE,
                axes=(1, 0),
                output_shapes=((3, 2), (3, 2)),
                split_axis=0,
            ),
            tuple(part.T for part in np.arange(12).reshape(2, 2, 3)),
        ),
    ],
)
def test_apply_transform_implements_the_closed_layout_set(
        source, transform, expected):
    """Every reviewed transform preserves the exact declared element ordering."""
    actual = apply_transform(source, transform)

    assert len(actual) == len(expected)
    assert all(
        np.array_equal(value, wanted)
        for value, wanted in zip(actual, expected))


def test_apply_transform_rejects_result_shape_mismatch():
    """A transform cannot publish a layout that differs from its declaration."""
    transform = Transform(TransformKind.TRANSPOSE,
                          axes=(1, 0),
                          output_shapes=((2, 3), ))

    with pytest.raises(ShapeMismatch, match="produced"):
        apply_transform(np.zeros((2, 3)), transform)


def test_convert_tensor_rounds_fp32_to_bf16_and_preserves_target_names():
    """FP32 halfway values use exact RNE bits in contiguous named BF16 output."""
    source_bits = np.asarray([
        0x3F800000,
        0x3F807FFF,
        0x3F808000,
        0x3F808001,
        0x3F817FFF,
        0x3F818000,
        0x3F818001,
        0xBF808000,
        0xBF818000,
    ],
                             dtype=np.uint32).reshape(3, 3)
    expected_bits = np.asarray([
        0x3F80,
        0x3F80,
        0x3F80,
        0x3F81,
        0x3F81,
        0x3F82,
        0x3F82,
        0xBF80,
        0xBF82,
    ],
                               dtype=np.uint16).reshape(3, 3)
    source = source_bits.view(np.float32)
    rule = MappingRule(
        source_path=("leaf", ),
        source_shape=(3, 3),
        target_names=("model.language_model.test.weight", ),
        target_shapes=((3, 3), ),
        transform=Transform(TransformKind.IDENTITY, output_shapes=((3, 3), )),
    )

    direct = convert_fp32_to_bf16(source)
    converted = convert_tensor(source, rule)
    tensor = converted[rule.target_names[0]]

    assert tuple(converted) == rule.target_names
    for result in (direct, tensor):
        assert result.dtype == torch.bfloat16
        assert result.device.type == "cpu"
        assert result.is_contiguous()
        assert np.array_equal(result.view(torch.uint16).numpy(), expected_bits)


def test_bf16_conversion_preserves_infinities_and_quiets_nans():
    """Special FP32 values remain signed infinities or BF16 NaNs."""
    source_bits = np.asarray([0x7F800000, 0xFF800000, 0x7F800001, 0xFF800001],
                             dtype=np.uint32)

    result = convert_fp32_to_bf16(source_bits.view(np.float32))
    result_bits = result.view(torch.uint16).numpy()

    assert np.array_equal(result_bits[:2],
                          np.asarray([0x7F80, 0xFF80], dtype=np.uint16))
    assert np.all((result_bits[2:] & np.uint16(0x7F80)) == np.uint16(0x7F80))
    assert np.all((result_bits[2:] & np.uint16(0x007F)) != 0)


@pytest.mark.parametrize(
    "source",
    [
        torch.zeros((2, 2), dtype=torch.bfloat16),
        np.zeros((2, 2), dtype=np.float16),
        np.zeros((2, 2), dtype=np.int32),
    ],
)
def test_convert_tensor_rejects_source_shape_and_dtype_mismatch(source):
    """Conversion rejects shape errors and BF16, FP16, or integer sources."""
    rule = MappingRule(
        source_path=("leaf", ),
        source_shape=(2, 2),
        target_names=("target", ),
        target_shapes=((2, 2), ),
        transform=Transform(TransformKind.IDENTITY, output_shapes=((2, 2), )),
    )

    with pytest.raises(ShapeMismatch, match="has shape"):
        convert_tensor(np.zeros((4, ), dtype=np.float32), rule)
    with pytest.raises(ShapeMismatch, match="FP32"):
        convert_tensor(source, rule)


def test_transform_contract_rejects_split_count_and_rule_descriptor_mismatch():
    """Split count and transform/target descriptors must agree before conversion."""
    split = Transform(TransformKind.SPLIT_TRANSPOSE_RESHAPE,
                      output_shapes=((2, ), ),
                      split_axis=0)
    with pytest.raises(ShapeMismatch, match="split produced"):
        apply_transform(np.zeros((2, 2)), split)

    rule = MappingRule(
        source_path=("leaf", ),
        source_shape=(2, ),
        target_names=("target", ),
        target_shapes=((2, ), ),
        transform=Transform(TransformKind.IDENTITY, output_shapes=((1, 2), )),
    )
    with pytest.raises(ShapeMismatch, match="transform declares"):
        convert_tensor(np.zeros((2, )), rule)
