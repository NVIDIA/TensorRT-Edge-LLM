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

from tensorrt_edgellm._native import dependencies, load


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


def test_runtime_extension_loads_before_global_plugin(monkeypatch, tmp_path):
    extension = tmp_path / "_edgellm_runtime.abi3.so"
    plugin = tmp_path / "libNvInfer_edgellm_plugin.so"
    resolved = load.ResolvedPayload("test-variant", extension, plugin)
    module = SimpleNamespace()
    plugin_handle = object()
    events = []

    monkeypatch.setattr(load, "_MODULE", None)
    monkeypatch.setattr(load, "_PLUGIN_HANDLE", None)
    monkeypatch.setattr(load, "_RETAINED_PLUGIN_HANDLES", [])
    monkeypatch.setattr(load, "resolve_payload", lambda **_kwargs: resolved)
    monkeypatch.setattr(load, "_existing_extension", lambda _path: None)
    monkeypatch.setattr(load, "_set_plugin_path", lambda _path: None)

    def import_extension(path):
        assert path == extension
        events.append("extension")
        return module

    def load_plugin(path, mode):
        assert path == str(plugin)
        assert mode
        events.append("plugin")
        return plugin_handle

    monkeypatch.setattr(load, "_import_extension", import_extension)
    monkeypatch.setattr(load.ctypes, "CDLL", load_plugin)

    assert load.load_runtime() is module
    assert events == ["extension", "plugin"]
    assert load._PLUGIN_HANDLE is plugin_handle
    assert load._RETAINED_PLUGIN_HANDLES == [plugin_handle]


def test_development_runtime_uses_one_module_for_both_import_names(
        monkeypatch, tmp_path):
    extension = tmp_path / "_edgellm_runtime.abi3.so"
    extension.touch()
    module = SimpleNamespace(__file__=str(extension))

    for module_name in load._EXTENSION_MODULE_NAMES:
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setitem(sys.modules, "tensorrt_edgellm._edgellm_runtime",
                        module)
    monkeypatch.setattr(load, "_MODULE", None)

    assert load.load_runtime_from_path(extension) is module
    assert sys.modules["_edgellm_runtime"] is module
    assert sys.modules["tensorrt_edgellm._edgellm_runtime"] is module
