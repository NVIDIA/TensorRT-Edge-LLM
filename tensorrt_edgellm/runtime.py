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
"""Lazy public facade for the packaged EdgeLLM native runtime."""

from types import ModuleType
from typing import Any, List

from ._native import (NativeDetectionError, NativeLoadError,
                      NativeManifestError, NativeRuntimeError,
                      NativeSelectionError)
from ._native.load import get_loaded_runtime, load_runtime


def load() -> ModuleType:
    """Detect, validate, and return the native runtime module.

    Returns:
        The selected ``_edgellm_runtime`` pybind11 module.

    Raises:
        NativeRuntimeError: If the installed wheel has no compatible payload or
            a payload cannot be loaded safely.
    """
    return load_runtime()


def __getattr__(name: str) -> Any:
    """Resolve public binding attributes on first access."""
    return getattr(load(), name)


def __dir__() -> List[str]:
    """Return facade and already-loaded binding attributes without native I/O."""
    module = get_loaded_runtime()
    binding_names = set(dir(module)) if module is not None else set()
    return sorted(set(globals()) | binding_names)


__all__ = [
    "NativeDetectionError",
    "NativeLoadError",
    "NativeManifestError",
    "NativeRuntimeError",
    "NativeSelectionError",
    "load",
]
