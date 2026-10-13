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
"""Load platform Python dependencies required by wheel build workflows."""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

from . import TensorRTDependencyError

_TENSORRT_HANDLE: Optional[ctypes.CDLL] = None

_INSTALL_GUIDANCE = (
    "Install the TensorRT Python package that matches the selected wheel "
    "payload, then retry. Qualified 0.11.0 packages are "
    "tensorrt==10.16.0.72 for TensorRT 10 payloads other than DGX Spark, "
    "tensorrt==10.16.1.11 for DGX Spark, and tensorrt==11.0.0.114 for "
    "TensorRT 11 x86 payloads. Install from PyPI with NVIDIA's package index; "
    "see the published-wheel section of the TensorRT Edge-LLM Installation "
    "Guide.")


def _runtime_soname(module: ModuleType) -> str:
    version = str(getattr(module, "__version__", ""))
    match = re.match(r"^(\d+)\.", version)
    if match is None:
        raise TensorRTDependencyError(
            "The installed TensorRT Python package has no parseable version "
            f"({version!r}). {_INSTALL_GUIDANCE}")
    return f"libnvinfer.so.{int(match.group(1))}"


def _packaged_runtime_path(soname: str) -> Optional[Path]:
    spec = importlib.util.find_spec("tensorrt_libs")
    if spec is None:
        return None
    roots = list(spec.submodule_search_locations or ())
    if spec.origin:
        roots.append(str(Path(spec.origin).parent))
    for root in roots:
        candidate = Path(root) / soname
        if candidate.is_file():
            return candidate
    return None


def _promote_runtime(module: ModuleType) -> str:
    """Expose pip-installed TensorRT libraries to later native DSO loads."""
    global _TENSORRT_HANDLE
    soname = _runtime_soname(module)
    if _TENSORRT_HANDLE is not None:
        return soname

    mode = getattr(os, "RTLD_GLOBAL", ctypes.RTLD_GLOBAL)
    mode |= getattr(os, "RTLD_NOW", 0)
    try:
        _TENSORRT_HANDLE = ctypes.CDLL(soname, mode=mode)
    except OSError as first_error:
        packaged = _packaged_runtime_path(soname)
        if packaged is None:
            raise TensorRTDependencyError(
                f"TensorRT {module.__version__} is installed, but {soname} "
                "cannot be loaded from the active Python environment. "
                f"{_INSTALL_GUIDANCE}") from first_error
        try:
            _TENSORRT_HANDLE = ctypes.CDLL(str(packaged), mode=mode)
        except OSError as error:
            raise TensorRTDependencyError(
                f"TensorRT {module.__version__} is installed, but its runtime "
                f"library {packaged} cannot be loaded: {error}. "
                f"{_INSTALL_GUIDANCE}") from error
    return soname


def require_tensorrt() -> ModuleType:
    """Import TensorRT and make a venv-packaged runtime globally loadable."""
    try:
        module = importlib.import_module("tensorrt")
    except (ImportError, OSError) as error:
        raise TensorRTDependencyError(
            "TensorRT Python bindings are required before checkpoint download "
            f"or engine build. {_INSTALL_GUIDANCE}") from error
    _promote_runtime(module)
    return module


def loaded_tensorrt_soname() -> Optional[str]:
    """Return the active Python TensorRT ABI, without importing it."""
    module = sys.modules.get("tensorrt")
    if module is None:
        return None
    return _promote_runtime(module)


__all__ = ["loaded_tensorrt_soname", "require_tensorrt"]
