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
"""Numerical boundaries of direct-builder AWQ repacking."""

import numpy as np
import pytest

from experimental.builder.weight_packing import int4


def _packed_words(nibble, rows):
    word = np.uint32(nibble * 0x11111111)
    return np.full((rows, 1), word, dtype=np.uint32).view(np.int32)


@pytest.mark.parametrize("plugin_version", [1, 2])
@pytest.mark.parametrize("q,zero", [(15, 3), (0, 15), (15, 0), (0, 9)])
def test_awq_repacking_rejects_unrepresentable_asymmetric_values(
        q, zero, plugin_version):
    qweight = _packed_words(q, 64)
    qzeros = _packed_words(zero, 1)
    with pytest.raises(ValueError, match="AWQ.*signed INT4"):
        int4.repack_awq(qweight, qzeros, plugin_version)


@pytest.mark.parametrize("plugin_version", [1, 2])
def test_awq_repacking_rejects_one_unrepresentable_group_value(plugin_version):
    qweight = _packed_words(8, 64)
    qzeros = _packed_words(8, 2)
    qweight[33, 0] = np.array(0x8888888F, dtype=np.uint32).view(np.int32)
    qzeros[1, 0] = _packed_words(3, 1)[0, 0]
    with pytest.raises(ValueError, match="AWQ.*signed INT4"):
        int4.repack_awq(qweight, qzeros, plugin_version)


@pytest.mark.parametrize("plugin_version", [1, 2])
@pytest.mark.parametrize("q,zero", [(0, 8), (15, 8), (3, 3), (8, 1), (7, 15)])
def test_awq_repacking_preserves_representable_values(q, zero, plugin_version):
    qweight = _packed_words(q, 64)
    qzeros = _packed_words(zero, 1)
    expected_nibbles = np.full((8, 64), q - zero + 8, dtype=np.int16)
    expected = int4._pack_for_gemm(expected_nibbles, plugin_version)
    np.testing.assert_array_equal(
        int4.repack_awq(qweight, qzeros, plugin_version), expected)
