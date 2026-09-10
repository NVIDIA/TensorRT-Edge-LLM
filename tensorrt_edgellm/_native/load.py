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
"""Validate and load exactly one installed EdgeLLM native payload."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Callable, Optional

from . import NativeLoadError
from .detect import DetectedPlatform, detect_platform
from .select import load_manifest, select_variant

_LOCK = threading.Lock()
_MODULE: Optional[ModuleType] = None
_PLUGIN_HANDLE: Optional[ctypes.CDLL] = None
_RETAINED_PLUGIN_HANDLES = []


@dataclass(frozen=True)
class ResolvedPayload:
    """Validated paths for one selected installed payload."""

    variant_id: str
    extension: Path
    plugin: Path


def _safe_payload_path(package_root: Path, relative_path: str) -> Path:
    manifest_path = PurePosixPath(relative_path)
    if manifest_path.is_absolute() or ".." in manifest_path.parts:
        raise NativeLoadError(
            f"Native payload path must be relative and traversal-free: {relative_path!r}."
        )
    candidate = package_root.joinpath(*manifest_path.parts)
    try:
        resolved_root = package_root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise NativeLoadError(
            f"Native payload path does not exist: {candidate}.") from error
    if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
        raise NativeLoadError(
            f"Native payload path escapes the installed package: {relative_path!r}."
        )
    if not resolved.is_file():
        raise NativeLoadError(
            f"Native payload is not a regular file: {resolved}.")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as payload:
        for block in iter(lambda: payload.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if not expected or actual.lower() != expected.lower():
        raise NativeLoadError(
            f"{label} SHA-256 mismatch for {path}: expected {expected}, found {actual}."
        )


def _set_plugin_path(plugin_path: Path) -> None:
    existing = os.environ.get("EDGELLM_PLUGIN_PATH")
    if existing:
        try:
            existing_path = Path(existing).resolve(strict=True)
        except OSError as error:
            raise NativeLoadError(
                f"EDGELLM_PLUGIN_PATH points to a missing file: {existing}."
            ) from error
        if existing_path != plugin_path:
            raise NativeLoadError(
                "EDGELLM_PLUGIN_PATH conflicts with the selected wheel payload: "
                f"{existing_path} != {plugin_path}. Unset it before starting Python."
            )
    os.environ["EDGELLM_PLUGIN_PATH"] = str(plugin_path)


def _existing_extension(extension_path: Path) -> Optional[ModuleType]:
    module_name = "tensorrt_edgellm._edgellm_runtime"
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file is None or Path(
                existing_file).resolve() != extension_path:
            raise NativeLoadError(
                f"Python module {module_name} is already loaded from "
                f"{existing_file!r}, not selected payload {extension_path}.")
    return existing


def _import_extension(extension_path: Path) -> ModuleType:
    existing = _existing_extension(extension_path)
    if existing is not None:
        return existing
    module_name = "tensorrt_edgellm._edgellm_runtime"
    spec = importlib.util.spec_from_file_location(module_name, extension_path)
    if spec is None or spec.loader is None:
        raise NativeLoadError(
            f"Cannot create an import specification for {extension_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def resolve_payload(
    *,
    package_root: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    detector: Callable[[],
                       DetectedPlatform] = detect_platform) -> ResolvedPayload:
    """Select and verify one payload without importing its extension."""
    root = (package_root or Path(__file__).resolve().parents[1]).resolve()
    manifest_file = manifest_path or root / "_native" / "variants.json"
    selected = select_variant(detector(), load_manifest(manifest_file))
    extension = _safe_payload_path(root, str(selected["extension"]))
    plugin = _safe_payload_path(root, str(selected["plugin"]))
    _verify_hash(extension, str(selected["extension_sha256"]), "Extension")
    _verify_hash(plugin, str(selected["plugin_sha256"]), "Plugin")
    return ResolvedPayload(
        variant_id=str(selected["variant_id"]),
        extension=extension,
        plugin=plugin,
    )


def load_runtime(
        *,
        package_root: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
        detector: Callable[[],
                           DetectedPlatform] = detect_platform) -> ModuleType:
    """Load and cache the native module selected for the current process."""
    global _MODULE, _PLUGIN_HANDLE
    with _LOCK:
        if _MODULE is not None:
            return _MODULE
        payload = resolve_payload(
            package_root=package_root,
            manifest_path=manifest_path,
            detector=detector,
        )
        _existing_extension(payload.extension)
        _set_plugin_path(payload.plugin)
        try:
            mode = getattr(os, "RTLD_GLOBAL", ctypes.RTLD_GLOBAL)
            mode |= getattr(os, "RTLD_NOW", 0)
            mode |= getattr(os, "RTLD_NODELETE", 0)
            plugin_handle = ctypes.CDLL(str(payload.plugin), mode=mode)
            _PLUGIN_HANDLE = plugin_handle
            _RETAINED_PLUGIN_HANDLES.append(plugin_handle)
            module = _import_extension(payload.extension)
        except Exception as error:
            raise NativeLoadError(
                f"Failed to load native payload {payload.variant_id}: {error}"
            ) from error
        _MODULE = module
        return module


def get_loaded_runtime() -> Optional[ModuleType]:
    """Return the loaded module without performing detection or native I/O."""
    return _MODULE
