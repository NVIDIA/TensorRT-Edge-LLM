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
"""Behavioral contracts for generic CLI dispatch."""

import subprocess
import sys

from experimental.checkpoint_converters import cli
from experimental.checkpoint_converters.schema import ConversionReport


def test_cli_module_help_is_warning_free():
    """Module help exposes common arguments without import warnings."""
    result = subprocess.run(
        [
            sys.executable, "-m", "experimental.checkpoint_converters.cli",
            "--help"
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--model-family" in result.stdout
    assert "--model-variant" in result.stdout
    assert "--checkpoint-path" in result.stdout
    assert "RuntimeWarning" not in result.stderr


def test_main_dispatches_selected_model_family(monkeypatch, tmp_path, capsys):
    """The selected registry converter builds and receives the common request."""
    checkpoint = tmp_path / "checkpoint"
    tokenizer = tmp_path / "tokenizer"
    output = tmp_path / "output"
    request = object()
    report = ConversionReport(output_path=output,
                              tensor_count=3,
                              checkpoint_files=("weights", ),
                              total_size=17)
    calls = []

    class Converter:

        def build_request(self, args):
            calls.append(("build", args.checkpoint_path, args.tokenizer_path,
                          args.output_path, args.model_variant))
            return request

        def convert(self, received):
            calls.append(("convert", received))
            return report

    converter = Converter()
    monkeypatch.setattr(cli, "get_supported_model_families", lambda:
                        ("test-family", ))
    monkeypatch.setattr(
        cli, "get_checkpoint_converter", lambda name: converter
        if name == "test-family" else None)

    assert cli.main([
        "--model-family", "test-family", "--model-variant", "test-variant",
        "--checkpoint-path",
        str(checkpoint), "--tokenizer-path",
        str(tokenizer), "--output-path",
        str(output)
    ]) == 0
    assert calls == [("build", checkpoint, tokenizer, output, "test-variant"),
                     ("convert", request)]
    assert "Published 3 tensors in 1 files" in capsys.readouterr().out
