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
"""Pure manifest parsing and exact native payload selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from . import (NativeManifestError, NativeManifestNotFoundError,
               NativeSelectionError)
from .contract import validate_manifest
from .detect import DetectedPlatform


def load_manifest(path: Path) -> Dict[str, Any]:
    """Read and validate an installed native manifest."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise NativeManifestNotFoundError(
            f"This installation has no native payload manifest at {path}. "
            "Install a released TensorRT Edge-LLM wheel for this architecture "
            "and Python version.") from error
    except (OSError, json.JSONDecodeError) as error:
        raise NativeManifestError(
            f"Cannot read native payload manifest {path}: {error}") from error
    try:
        return validate_manifest(value)
    except ValueError as error:
        raise NativeManifestError(str(error)) from error


def _probe_matches(entry: Mapping[str, Any],
                   detected: DetectedPlatform) -> bool:
    if entry["platform_probe_source"] != detected.platform_probe_source:
        return False
    candidates = entry["platform_probe_values"]
    if detected.platform_probe_source == "device-model":
        model = detected.platform_probe_value.casefold()
        return any(
            str(candidate).casefold() in model for candidate in candidates)
    return detected.platform_probe_value in candidates


def _matches(entry: Mapping[str, Any], detected: DetectedPlatform) -> bool:
    return (entry["python_abi"] == detected.python_abi
            and entry["cpu_arch"] == detected.cpu_arch
            and entry["cuda_runtime_soname"] == detected.cuda_runtime_soname
            and entry["tensorrt_runtime_soname"]
            == detected.tensorrt_runtime_soname
            and int(entry["gpu_sm"]) == detected.gpu_sm
            and _probe_matches(entry, detected))


def _supported_rows(entries: Sequence[Mapping[str, Any]],
                    detected: DetectedPlatform) -> str:
    relevant = [
        str(entry.get("variant_id", "<unnamed>")) for entry in entries
        if entry.get("cpu_arch") == detected.cpu_arch
        and entry.get("python_abi") == detected.python_abi
    ]
    return ", ".join(
        sorted(relevant)) if relevant else "none for this architecture/ABI"


def select_variant(detected: DetectedPlatform,
                   manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the unique payload matching all detected facts."""
    entries = manifest.get("variants")
    if not isinstance(entries, list):
        raise NativeManifestError(
            "Native payload manifest has no variants list.")
    matches = [entry for entry in entries if _matches(entry, detected)]
    if len(matches) != 1:
        raise NativeSelectionError(
            "Expected one exact TensorRT Edge-LLM native payload, found "
            f"{len(matches)} for {detected.as_dict()}. Supported rows: "
            f"{_supported_rows(entries, detected)}.")
    return dict(matches[0])
