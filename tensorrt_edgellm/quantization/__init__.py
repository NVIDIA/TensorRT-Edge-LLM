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
"""Standalone quantization for TensorRT Edge-LLM.

Decoupled from the ONNX exporter — runs in a clean venv with only
torch, transformers, and modelopt.

    tensorrt-edgellm-quantize --help
"""

_EXPORT_MODULES = {
    "DFlashCalibDraftModel": ".models",
    "Eagle3DraftModel": ".models",
    "quantize_and_export_dflash_draft": ".models",
    "quantize_and_export_draft": ".models",
    "build_quant_config": ".quantization_configs",
    "quantize_and_export": ".quantize",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name):
    import importlib

    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_EXPORT_MODULES[name], __name__),
                    name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
