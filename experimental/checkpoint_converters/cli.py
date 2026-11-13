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
"""Command-line entry point for Flax checkpoint converters."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .models import get_checkpoint_converter, get_supported_model_families


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(
        "Convert a supported Orbax checkpoint to the HF-style checkpoint "
        "consumed by TensorRT Edge-LLM."))
    parser.add_argument(
        "--model-family",
        required=True,
        choices=get_supported_model_families(),
        help="model family converter to run",
    )
    parser.add_argument(
        "--model-variant",
        help=
        "exact model variant; defaults to the family's first supported variant",
    )
    parser.add_argument("--checkpoint-path",
                        required=True,
                        type=Path,
                        help="local Orbax checkpoint root")
    parser.add_argument(
        "--tokenizer-path",
        required=True,
        type=Path,
        help="local directory with the family-required tokenizer assets")
    parser.add_argument("--output-path",
                        required=True,
                        type=Path,
                        help="absent or empty output directory")
    parser.add_argument(
        "--restore-concurrent-gb",
        type=int,
        default=16,
        help="Orbax restore concurrency bound in GB (default: 16)")
    parser.add_argument(
        "--max-checkpoint-file-bytes",
        type=int,
        default=5 * 1024**3,
        help="planned tensor-byte bound per safetensors file (default: 5 GiB)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run conversion, and print a concise report.

    Args:
        argv: Optional argument sequence excluding the executable name.

    Returns:
        Zero after successful atomic publication.
    """
    args = _parser().parse_args(argv)
    converter = get_checkpoint_converter(args.model_family)
    request = converter.build_request(args)
    report = converter.convert(request)
    print(f"Published {report.tensor_count} tensors in "
          f"{len(report.checkpoint_files)} files to {report.output_path} "
          f"({report.total_size} uncompressed bytes).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
