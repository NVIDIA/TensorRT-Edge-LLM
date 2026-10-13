#!/usr/bin/env python3
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

"""Shared helpers for split FC1/FC2 NVFP4 MoE exports."""

from __future__ import annotations

from pathlib import Path


SF_VEC_SIZE = 16
M_TILE_SIZE = 128


def resolve_output_dtype(name):
    import cutlass

    mapping = {
        "bf16": cutlass.BFloat16,
        "fp16": cutlass.Float16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported output_dtype {name!r}; expected one of {sorted(mapping)}")
    return mapping[name]


def resolve_activation_type(name):
    from moe_compat import ActivationType

    mapping = {
        "relu2": ActivationType.Relu2,
        "swiglu": ActivationType.Swiglu,
        "geglu": ActivationType.Geglu,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation {name!r}; expected one of {sorted(mapping)}")
    return mapping[name]


def verify_export(output_dir: str, file_name: str) -> tuple[str, str]:
    header = Path(output_dir) / f"{file_name}.h"
    obj = Path(output_dir) / f"{file_name}.o"
    missing = [str(path) for path in (header, obj) if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"CuTeDSL export did not produce non-empty artifacts: {missing}")
    return str(header), str(obj)
