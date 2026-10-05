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
"""Shared helpers for the FP16 MoE grouped-GEMM AOT export."""

from __future__ import annotations

import pathlib

from cutedsl_utils import aot_placeholders

# Max expert count: sizes the AOT trace buffers and caps the persistent grid.
# Not baked into the cubin -- the runtime group_count argument carries E.
MAX_NUM_EXPERTS = 256
DESCRIPTOR_ALIGNMENT = 16
TENSORMAP_ALIGNMENT = 128
BYTES_PER_TENSORMAP = 128
TENSORMAPS_PER_BLOCK = 3


def runtime_max_active_clusters(family: str):
    """Create the persistent-grid scalar supplied by the runtime C++ caller.

    The exported wrapper receives the deployment GPU's persistent block count
    at runtime, so this value only establishes the AOT argument type. Do not
    probe the build GPU here (``HardwareInfo``): artifact generation must not
    depend on a physical GPU, and the occupancy probe launches a helper kernel
    that fails when cross-compiling a CTA-cluster target on a foreign GPU.
    """
    if family not in ("ampere", "blackwell", "blackwell_geforce"):
        raise ValueError(f"Unsupported f16_moe family: {family}")
    return aot_placeholders.runtime_int32()


def make_ptr(data_type, assumed_align: int = DESCRIPTOR_ALIGNMENT):
    """Build a typed, aligned pointer carrying no backing storage."""
    return aot_placeholders.make_ptr(data_type, assumed_align=assumed_align)


def verify_export(output_dir: str, file_name: str) -> tuple[str, str]:
    """Require a CuTeDSL export to produce nonempty header and object files."""
    header = pathlib.Path(output_dir) / f"{file_name}.h"
    obj = pathlib.Path(output_dir) / f"{file_name}.o"
    missing = [
        str(path) for path in (header, obj)
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            f"CuTeDSL export did not produce nonempty artifacts: {missing}")
    return str(header), str(obj)
