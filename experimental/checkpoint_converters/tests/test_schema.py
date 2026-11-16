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
"""Behavioral contracts for family-independent conversion requests."""

from dataclasses import dataclass

import pytest

from experimental.checkpoint_converters.schema import (ConversionError,
                                                       ConversionRequest)


@dataclass(frozen=True)
class _SyntheticRequest(ConversionRequest):
    family_error: bool = False

    def validate_family(self) -> None:
        if self.family_error:
            raise ConversionError("family validation failed")


def _paths(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    return checkpoint, tokenizer, tmp_path / "output"


def test_request_validates_common_paths_bounds_and_family_contract(tmp_path):
    """Common validation runs before the inherited family contract."""
    checkpoint, tokenizer, output = _paths(tmp_path)
    _SyntheticRequest(checkpoint, tokenizer, output).validate()

    with pytest.raises(ConversionError, match="family validation failed"):
        _SyntheticRequest(checkpoint, tokenizer, output,
                          family_error=True).validate()
    with pytest.raises(ConversionError, match="must be positive"):
        _SyntheticRequest(checkpoint,
                          tokenizer,
                          output,
                          restore_concurrent_gb=0).validate()


def test_request_preserves_existing_nonempty_output(tmp_path):
    """Common validation rejects an occupied destination without modification."""
    checkpoint, tokenizer, output = _paths(tmp_path)
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("original", encoding="utf-8")

    with pytest.raises(ConversionError, match="absent or empty"):
        _SyntheticRequest(checkpoint, tokenizer, output).validate()

    assert marker.read_text(encoding="utf-8") == "original"
