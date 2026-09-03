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
"""Validate a wheel source checkout and emit source provenance."""

import argparse
from pathlib import Path

from .config import (REPO_ROOT, require_clean_source, source_snapshot,
                     write_json)


def _arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args(argv)


def _emit_snapshot(root: Path, output: Path | None) -> None:
    snapshot = source_snapshot(root)
    if output:
        write_json(output, snapshot)
    else:
        print(snapshot["sha256"])


def main(argv=None) -> None:
    """Optionally require a clean checkout and emit its source snapshot."""
    args = _arguments(argv)
    root = args.repo_root.resolve()
    if args.require_clean:
        require_clean_source(root)
    _emit_snapshot(root, args.output)


if __name__ == "__main__":
    main()
