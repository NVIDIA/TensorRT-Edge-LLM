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
"""Tests for GPTQ MoE asymmetric weight conversion."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tensorrt_edgellm.checkpoint.repacking import _extract_gptq_for_marlin


@pytest.mark.parametrize("weight,zero", [(15, 3), (0, 12)])
def test_moe_rejects_out_of_range_asymmetric_values(weight, zero):
    packed_weight = weight * 0x11111111
    if packed_weight >= 2**31:
        packed_weight -= 2**32
    packed_zero = zero * 0x11111111
    if packed_zero >= 2**31:
        packed_zero -= 2**32
    projection = SimpleNamespace(
        qweight=torch.full((16, 16), packed_weight, dtype=torch.int32),
        qzeros=torch.full((2, 2), packed_zero, dtype=torch.int32),
        scales=torch.ones(2, 16),
    )
    with pytest.raises(ValueError, match="cannot represent"):
        _extract_gptq_for_marlin(projection, 64, zero_point_offset=0)


@pytest.mark.parametrize("weight,zero", [(10, 3), (0, 8)])
def test_moe_preserves_representable_asymmetric_values(weight, zero):
    packed_weight = weight * 0x11111111
    if packed_weight >= 2**31:
        packed_weight -= 2**32
    packed_zero = zero * 0x11111111
    if packed_zero >= 2**31:
        packed_zero -= 2**32
    projection = SimpleNamespace(
        qweight=torch.full((16, 16), packed_weight, dtype=torch.int32),
        qzeros=torch.full((2, 2), packed_zero, dtype=torch.int32),
        scales=torch.ones(2, 16),
    )
    weights, scales = _extract_gptq_for_marlin(projection,
                                               64,
                                               zero_point_offset=0)
    assert torch.all((weights - 8) *
                     scales.repeat_interleave(64, dim=1) == weight - zero)
