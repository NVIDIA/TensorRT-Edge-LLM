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
"""CLI version reporting without loading the native runtime."""

import argparse
import pathlib
import subprocess
from importlib import metadata

SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[1]


def version_string():
    """Use installed metadata, with a source fallback and optional checkout revision."""
    try:
        package_version = metadata.version("tensorrt-edgellm")
    except metadata.PackageNotFoundError:
        from tensorrt_edgellm._version import __version__
        package_version = __version__
    result = f"edge-llm {package_version}"
    if (SOURCE_ROOT / ".git").exists():
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "--short=9", "HEAD"],
                cwd=SOURCE_ROOT,
                capture_output=True,
                text=True,
                check=True,
                timeout=2).stdout.strip()
            if revision:
                result += f" (git {revision})"
        except (OSError, subprocess.SubprocessError):
            pass
    return result


class VersionAction(argparse.Action):
    """Resolve the version only when requested."""

    def __call__(self, parser, namespace, values, option_string=None):
        print(version_string())
        parser.exit()
