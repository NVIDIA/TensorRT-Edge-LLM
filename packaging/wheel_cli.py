#!/usr/bin/env python3
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
"""Build and verify TensorRT Edge-LLM wheels from source."""

from __future__ import annotations

import subprocess
import sys
from typing import Callable, Dict, Optional, Sequence

from wheellib import assemble, base, build, cutedsl, payload, source, verify
from wheellib.config import REPO_ROOT, load_matrix


def _validate_matrix(_: Sequence[str]) -> None:
    load_matrix(REPO_ROOT / "packaging" / "variants.toml")


def main(values: Optional[Sequence[str]] = None) -> int:
    """Dispatch one supported public packaging command."""
    arguments = list(values if values is not None else sys.argv[1:])
    commands: Dict[str, Callable[[Sequence[str]], None]] = {
        "validate-source": source.main,
        "validate-matrix": _validate_matrix,
        "build-wheel": build.main,
        "build-base": base.main,
        "prepare-cutedsl": cutedsl.main,
        "build-payload": payload.main,
        "verify-payload": verify.main,
        "assemble": assemble.main,
    }
    if not arguments or arguments[0] not in commands:
        available = ", ".join(sorted(commands))
        raise RuntimeError(
            f"Usage: python packaging/wheel_cli.py <command> [args]\n"
            f"Commands: {available}")
    commands[arguments[0]](arguments[1:])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError,
            subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
