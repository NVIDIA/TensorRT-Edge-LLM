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
"""Tests for checkpoint destination dtype preservation."""

import torch
import torch.nn as nn

from tensorrt_edgellm.checkpoint.loader import _set_tensor


class _TinyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.fp16 = nn.Linear(3, 2, dtype=torch.float16)
        self.bf16 = nn.Linear(3, 2, dtype=torch.bfloat16)
        self.fp32 = nn.Linear(3, 2, dtype=torch.float32)
        self.register_buffer("integer", torch.zeros(2, dtype=torch.int32))
        self.attribute = torch.zeros(2, dtype=torch.float32)


def test_fp32_source_cast_to_declared_fp16_destination():
    model = _TinyModel()

    assert _set_tensor(model, "fp16.weight",
                       torch.ones_like(model.fp16.weight, dtype=torch.float32))

    assert model.fp16.weight.dtype == torch.float16


def test_fp32_source_cast_to_declared_bf16_destination():
    model = _TinyModel()

    assert _set_tensor(model, "bf16.weight",
                       torch.ones_like(model.bf16.weight, dtype=torch.float32))

    assert model.bf16.weight.dtype == torch.bfloat16


def test_matching_half_dtype_is_preserved():
    model = _TinyModel()

    assert _set_tensor(model, "bf16.weight",
                       torch.ones_like(model.bf16.weight))

    assert model.bf16.weight.dtype == torch.bfloat16


def test_bf16_source_cast_to_declared_fp16_destination():
    model = _TinyModel()

    assert _set_tensor(
        model, "fp16.weight",
        torch.ones_like(model.fp16.weight, dtype=torch.bfloat16))

    assert model.fp16.weight.dtype == torch.float16


def test_fp32_destination_keeps_fp32_source():
    model = _TinyModel()

    assert _set_tensor(model, "fp32.weight",
                       torch.ones_like(model.fp32.weight))

    assert model.fp32.weight.dtype == torch.float32


def test_unknown_destination_retains_legacy_bf16_fallback():
    model = _TinyModel()

    assert _set_tensor(model, "attribute", torch.ones(2, dtype=torch.bfloat16))

    assert model.attribute.dtype == torch.float16


def test_non_floating_source_is_not_cast():
    model = _TinyModel()

    assert _set_tensor(model, "integer", torch.ones(2, dtype=torch.int64))

    assert model.integer.dtype == torch.int64
