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
"""Concise dependency errors for optional command-line tools."""

import functools
import shlex
import sys

from tensorrt_edgellm import __version__, _dependencies


def optional_dependencies(extra):
    """Report missing optional packages without hiding other import errors."""

    def decorate(main):

        @functools.wraps(main)
        def run(*args, **kwargs):
            try:
                return main(*args, **kwargs)
            except ModuleNotFoundError as exc:
                known = _dependencies.EXPORT_DEPENDENCIES if extra == "export" else _dependencies.TOOL_DEPENDENCIES
                if exc.name not in known:
                    raise
                raise SystemExit(
                    f"Missing optional dependency '{exc.name}'. "
                    f"Install dependencies with: {shlex.quote(sys.executable)} -m pip install "
                    f'"tensorrt-edgellm[{extra}]=={__version__}"') from None

        return run

    return decorate
