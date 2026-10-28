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
"""pi0.5 (openpi VLA) model/export support.

The Python model definitions and export contracts live in ``tensorrt_edgellm``.
The experimental C++ runtime remains under ``experimental_models/pi05`` and
consumes the component artifacts emitted by this package.
"""

_EXPORT_MODULES = {
    "Pi05ActionConfig": ".modeling_pi05_action",
    "Pi05PrefixConfig": ".modeling_pi05_prefix",
    "Pi05VisualConfig": ".modeling_pi05_visual",
    "build_pi05_action": ".modeling_pi05_action",
    "build_pi05_prefix": ".modeling_pi05_prefix",
    "build_pi05_visual": ".modeling_pi05_visual",
    "is_pi05_checkpoint": ".weights",
    "is_pi05_weights": ".weights",
    "load_checkpoint_weights": ".weights",
    "load_pi05_config": ".weights",
    "split_pi05_weights": ".weights",
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
