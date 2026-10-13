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

import sys
from types import SimpleNamespace

from tensorrt_edgellm._native import dependencies


def test_pip_tensorrt_runtime_is_loaded_without_system_registration(
        monkeypatch, tmp_path):
    module = SimpleNamespace(__version__="10.16.0.72")
    runtime = tmp_path / "tensorrt_libs" / "libnvinfer.so.10"
    runtime.parent.mkdir()
    runtime.touch()
    attempts = []
    handle = object()

    def load(path, mode):
        attempts.append(path)
        if path == "libnvinfer.so.10":
            raise OSError("not in the system linker cache")
        return handle

    monkeypatch.setattr(dependencies, "_TENSORRT_HANDLE", None)
    monkeypatch.setattr(dependencies.importlib, "import_module",
                        lambda _name: module)
    monkeypatch.setitem(sys.modules, "tensorrt", module)
    monkeypatch.setattr(
        dependencies.importlib.util,
        "find_spec",
        lambda _name: SimpleNamespace(
            submodule_search_locations=[str(runtime.parent)], origin=None),
    )
    monkeypatch.setattr(dependencies.ctypes, "CDLL", load)

    assert dependencies.require_tensorrt() is module
    assert attempts == ["libnvinfer.so.10", str(runtime)]
    assert dependencies.loaded_tensorrt_soname() == "libnvinfer.so.10"
    assert dependencies._TENSORRT_HANDLE is handle
