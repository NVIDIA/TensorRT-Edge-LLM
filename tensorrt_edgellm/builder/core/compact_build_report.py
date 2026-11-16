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
"""Whether an engine was built for compact execution.

Each compact-capable plugin layer sizes its compact workspace when the engine
is built. The plugin library decides from its build report whether every such
layer reserved it; only then may the engine run compact steps (prompt chunks
and decode rows in one step), and the builder sets
``builder_config.compact_execution`` to true.
"""

import ctypes
import json
import logging
from typing import Callable, Tuple, TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def build_with_compact_report(plugin_handle,
                              build: Callable[[], _T]) -> Tuple[_T, bool]:
    """Run ``build`` inside a compact build report of ``plugin_handle``.

    Returns the result of ``build`` and whether the plugin library found the
    engine built for compact execution (the library logs why not); False when
    the library has no report.
    """
    begin = getattr(plugin_handle, "edgellmBeginCompactBuildReport", None)
    end = getattr(plugin_handle, "edgellmEndCompactBuildReport", None)
    if begin is None or end is None:
        logger.warning("The plugin library has no compact build report; the "
                       "engine will run without compact steps")
        return build(), False
    begin.restype = ctypes.c_uint64
    end.restype = ctypes.c_bool
    end.argtypes = [ctypes.c_uint64]
    report_id = begin()
    try:
        result = build()
    finally:
        compact_execution = bool(end(report_id))
    return result, compact_execution


def write_compact_execution(config_path: str) -> None:
    """Set ``builder_config.compact_execution`` of an engine config to true."""
    with open(config_path) as config_file:
        config = json.load(config_file)
    config["builder_config"]["compact_execution"] = True
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=2)
