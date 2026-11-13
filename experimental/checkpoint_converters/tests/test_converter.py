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
"""Behavioral contracts for the generic converter boundary and registry."""

from argparse import Namespace
from dataclasses import dataclass

import pytest

from experimental.checkpoint_converters.converter import CheckpointConverter
from experimental.checkpoint_converters.models import (
    get_checkpoint_converter, get_supported_model_families)
from experimental.checkpoint_converters.models.gemma4.converter import \
    Gemma4CheckpointConverter
from experimental.checkpoint_converters.schema import (ConversionReport,
                                                       ConversionRequest,
                                                       UnsupportedConversion)


@dataclass(frozen=True)
class _Request(ConversionRequest):

    def validate_family(self):
        return None


class _Converter(CheckpointConverter):
    request_type = _Request

    def __init__(self, report):
        self.report = report
        self.received = None

    def convert(self, request):
        self.received = request
        return self.report


def test_converter_preserves_all_common_request_fields(tmp_path):
    """Request construction preserves every family-independent CLI value."""
    report = ConversionReport(tmp_path / "out", 1, ("weights", ), 2)
    converter = _Converter(report)
    args = Namespace(checkpoint_path=tmp_path / "checkpoint",
                     tokenizer_path=tmp_path / "tokenizer",
                     output_path=tmp_path / "out",
                     restore_concurrent_gb=3,
                     max_checkpoint_file_bytes=17,
                     model_variant=None)

    request = converter.build_request(args)
    result = converter.convert(request)

    assert request == _Request(args.checkpoint_path, args.tokenizer_path,
                               args.output_path, 3, 17)
    assert converter.received is request
    assert result is report


def test_registry_exposes_only_supported_family_converters():
    """The registry resolves Gemma4 and rejects unknown family names."""
    assert get_supported_model_families() == ("gemma4", )
    assert isinstance(get_checkpoint_converter("gemma4"),
                      Gemma4CheckpointConverter)
    with pytest.raises(UnsupportedConversion, match="unknown.*gemma4"):
        get_checkpoint_converter("unknown")
