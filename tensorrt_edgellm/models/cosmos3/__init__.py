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
"""Cosmos3-Omni model/export support.

The Python model definitions and export contracts live in ``tensorrt_edgellm``.
The experimental C++ runtime remains under ``experimental_models/cosmos3`` and
consumes the component artifacts emitted by this package.
"""

_EXPORT_MODULES = {
    "DEFAULT_ACTION_CHUNK_SIZE": ".policy_geometry",
    "DEFAULT_DOMAIN": ".policy_geometry",
    "DEFAULT_NUM_FRAMES": ".policy_geometry",
    "DEFAULT_NUM_INFERENCE_STEPS": ".modeling_gen",
    "DEFAULT_RAW_ACTION_DIM": ".policy_geometry",
    "Cosmos3GenConfig": ".modeling_gen",
    "Cosmos3PolicyGeometry": ".policy_geometry",
    "build_cosmos3_gen": ".modeling_gen",
    "build_cosmos3_und_prefill": ".modeling_und_prefill",
    "build_cosmos3_vae_encoder": ".modeling_vae_encoder",
    "get_vae_onnx_export_args": ".modeling_vae_encoder",
    "load_config_json": ".weights",
    "split_transformer_weights": ".weights",
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
