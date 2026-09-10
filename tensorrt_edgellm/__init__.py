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
"""TensorRT Edge-LLM Python package.

The checkpoint exporter is optional. Its PyTorch and ONNX modules are loaded
only when an export API is requested, so the checkpoint-direct builder and
server can import the package without installing the legacy export toolchain.
"""

from importlib import import_module
from typing import Any

from ._version import __version__

_EXPORT_API_NAMES = (
    "AutoModel",
    "ModelConfig",
    "QuantConfig",
    "export_onnx",
    "load_checkpoint_config_dicts",
    "load_config_dict",
    "load_weights",
    "register_model",
)
_EXPORT_API_NAME_SET = frozenset(_EXPORT_API_NAMES)
_EXPORT_DEPENDENCIES = frozenset({
    "onnx",
    "onnx_graphsurgeon",
    "onnxscript",
    "safetensors",
    "torch",
    "transformers",
})


def _load_export_api():
    try:
        module = import_module("._export_api", __name__)
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".", 1)[0]
        if missing in _EXPORT_DEPENDENCIES:
            raise ModuleNotFoundError(
                f"{exc}. Install TensorRT Edge-LLM with the 'export' extra "
                "to use the PyTorch/ONNX exporter.") from exc
        raise
    for name in _EXPORT_API_NAMES:
        globals()[name] = getattr(module, name)
    return module


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_API_NAME_SET:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(_load_export_api(), name)


def __dir__():
    return sorted(set(globals()) | _EXPORT_API_NAME_SET)


__all__ = ["__version__", *_EXPORT_API_NAMES]
