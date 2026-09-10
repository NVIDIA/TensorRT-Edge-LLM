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
"""Shared contract for build matrix rows and installed native payloads."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
QUALIFIED_PYTHON_ABIS = ("cp310", "cp311", "cp312")
MATRIX_VARIANT_FIELDS = frozenset({
    "variant_id",
    "platform_family",
    "platform_release",
    "platform_probe_source",
    "platform_probe_values",
    "cpu_arch",
    "cuda_runtime_soname",
    "tensorrt_runtime_soname",
    "gpu_sm",
    "cute_dsl_artifact_tag",
    "cuda_ctk_version",
    "embedded_target",
    "cmake_args",
})
RUNTIME_VARIANT_FIELDS = frozenset({
    "variant_id",
    "python_abi",
    "platform_family",
    "platform_release",
    "platform_probe_source",
    "platform_probe_values",
    "cpu_arch",
    "cuda_runtime_soname",
    "tensorrt_runtime_soname",
    "gpu_sm",
    "extension",
    "plugin",
    "extension_sha256",
    "plugin_sha256",
    "source_revision",
    "cutedsl_metadata_sha256",
})
PROBE_SOURCES = frozenset({"os-release", "l4t", "driveos", "device-model"})


def _required(mapping: Mapping[str, Any], fields: Iterable[str],
              label: str) -> None:
    missing = sorted(set(fields) - set(mapping))
    if missing:
        raise ValueError(
            f"{label} is missing required fields: {', '.join(missing)}.")


def validate_matrix_variant(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize one source matrix variant."""
    if not isinstance(value, Mapping):
        raise ValueError("Variant row must be a table.")
    _required(value, MATRIX_VARIANT_FIELDS, "Variant row")
    unexpected = sorted(set(value) - MATRIX_VARIANT_FIELDS)
    if unexpected:
        raise ValueError(
            f"Variant row contains unsupported fields: {', '.join(unexpected)}."
        )
    row = dict(value)
    if row["platform_probe_source"] not in PROBE_SOURCES:
        raise ValueError(
            f"Variant {row['variant_id']} has an invalid platform probe source."
        )
    probes = row["platform_probe_values"]
    if not isinstance(probes, list) or not probes or any(
            not isinstance(item, str) or not item for item in probes):
        raise ValueError(
            f"Variant {row['variant_id']} must declare platform probe values.")
    if row["cpu_arch"] not in {"x86_64", "aarch64"}:
        raise ValueError(
            f"Variant {row['variant_id']} has an invalid CPU architecture.")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]+", str(row["variant_id"])):
        raise ValueError(f"Invalid variant_id {row['variant_id']!r}.")
    return row


def validate_runtime_variant(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize one installed runtime variant."""
    if not isinstance(value, Mapping):
        raise ValueError("Runtime variant must be an object.")
    _required(value, RUNTIME_VARIANT_FIELDS, "Runtime variant")
    row = dict(value)
    if row["platform_probe_source"] not in PROBE_SOURCES:
        raise ValueError(
            f"Runtime variant {row['variant_id']} has an invalid probe source."
        )
    probes = row["platform_probe_values"]
    if not isinstance(probes, list) or not probes or any(
            not isinstance(item, str) or not item for item in probes):
        raise ValueError(
            f"Runtime variant {row['variant_id']} has invalid probe values.")
    if row["python_abi"] not in QUALIFIED_PYTHON_ABIS:
        raise ValueError(f"Unsupported Python ABI {row['python_abi']!r}.")
    if row["cpu_arch"] not in {"x86_64", "aarch64"}:
        raise ValueError(
            f"Runtime variant {row['variant_id']} has an invalid architecture."
        )
    for field, width in (("source_revision", 40), ("extension_sha256", 64),
                         ("plugin_sha256", 64), ("cutedsl_metadata_sha256",
                                                 64)):
        if not re.fullmatch(rf"[0-9a-f]{{{width}}}", str(row[field])):
            raise ValueError(f"Runtime variant has an invalid {field}.")
    return row


def validate_manifest(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a complete installed native manifest."""
    if (not isinstance(value, Mapping)
            or value.get("schema_version") != SCHEMA_VERSION):
        raise ValueError(
            f"Native manifest must use schema_version {SCHEMA_VERSION}.")
    variants = value.get("variants")
    if (not isinstance(variants, Sequence)
            or isinstance(variants, (str, bytes)) or not variants):
        raise ValueError("Native manifest contains no variants.")
    normalized = [validate_runtime_variant(row) for row in variants]
    identities = set()
    for row in normalized:
        identity = (
            row["python_abi"],
            row["platform_probe_source"],
            tuple(row["platform_probe_values"]),
            row["cpu_arch"],
            row["cuda_runtime_soname"],
            row["tensorrt_runtime_soname"],
            int(row["gpu_sm"]),
        )
        if identity in identities:
            raise ValueError(
                f"Duplicate runtime identity for {row['variant_id']}.")
        identities.add(identity)
    result = dict(value)
    result["variants"] = normalized
    return result
