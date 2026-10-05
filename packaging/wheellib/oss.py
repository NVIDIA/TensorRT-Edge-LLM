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
"""Stage policy-sanitized wheel sources and audit distributable bytes."""

from __future__ import annotations

import fnmatch
import hashlib
import importlib.util
import json
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from . import config

BLOCK_SIZE = 1024 * 1024


def enabled() -> bool:
    """Whether this checkout carries the internal release policy."""
    return False


def policy(root: Path = config.REPO_ROOT) -> dict:
    """Load the repository's mandatory OSS policy; never fail open."""
    if not enabled():
        return {"forbidden_patterns": []}
    return json.loads((root / "oss_release_manifest.json").read_text(
        encoding="utf-8"))["release_policy"]


def policy_digest(root: Path = config.REPO_ROOT) -> str:
    """Bind artifacts to both policy inputs and the sanitizer implementation."""
    if not enabled():
        return "0" * 64
    names = (
        "oss_release_manifest.json",
        "scripts/strip_internal_release.py",
        "scripts/check_oss_release_sanitizer.py",
        "packaging/wheellib/oss.py")
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode() + b"\0")
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def stage_source(root: Path,
                 destination: Path,
                 *,
                 include_submodules: bool = True) -> Path:
    """Copy tracked sources, then sanitize without modifying the checkout."""
    if not enabled():
        return root
    if destination.exists():
        raise RuntimeError(
            f"OSS source destination must not exist: {destination}.")
    script = root / "scripts/check_oss_release_sanitizer.py"
    spec = importlib.util.spec_from_file_location("_wheel_source_sanitizer",
                                                  script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load source sanitizer: {script}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination.mkdir(parents=True)
    try:
        module.stage_oss_source(root, destination, include_submodules)
    except Exception:
        shutil.rmtree(destination)
        raise
    return destination


def tree_digest(root: Path) -> str:
    """Hash a staged tree including relative names, content and symlink targets."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            data = path.readlink().as_posix().encode()
        elif path.is_file():
            data = bytes.fromhex(config.sha256(path))
        else:
            continue
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(data)
    return digest.hexdigest()


def audit_stream(stream: BinaryIO,
                 name: str,
                 root: Path = config.REPO_ROOT) -> None:
    """Check every byte, including native symbols and embedded source strings."""
    rules = policy(root)
    markers = [
        value.encode() for value in rules["forbidden_patterns"] +
        rules.get("wheel_forbidden_patterns", [])
    ]
    overlap = max(map(len, markers), default=1) - 1
    tail = b""
    while chunk := stream.read(BLOCK_SIZE):
        data = tail + chunk
        for marker in markers:
            if marker in data:
                raise RuntimeError(
                    f"{name} contains internal marker {marker.decode()!r}.")
        tail = data[-overlap:] if overlap else b""


def audit_file(path: Path, root: Path = config.REPO_ROOT) -> None:
    """Audit one native library or generated kernel input."""
    with path.open("rb") as stream:
        audit_stream(stream, str(path), root)


def _forbidden_path(name: str, root: Path) -> bool:
    if not enabled():
        return False
    patterns = list(policy(root).get("delete_paths", []))
    manifest = root / policy(root)["do_not_release_manifest"]
    patterns.extend(line.strip() for line in manifest.read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#"))
    parts = PurePosixPath(name).parts
    prefixes = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
    for pattern in patterns:
        normalized = pattern.rstrip("/")
        if "/" not in normalized and (pattern.endswith("/")
                                      or any(c in normalized for c in "*?[")):
            if any(fnmatch.fnmatchcase(part, normalized) for part in parts):
                return True
        elif any(
                fnmatch.fnmatchcase(prefix, normalized)
                for prefix in prefixes):
            return True
    return False


def audit_archive(archive: zipfile.ZipFile,
                  root: Path = config.REPO_ROOT) -> None:
    """Reject forbidden paths and internal bytes in every wheel member."""
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise RuntimeError("Wheel contains duplicate archive members.")
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if (path.is_absolute() or ".." in path.parts or "\\" in info.filename
                or stat.S_ISLNK(info.external_attr >> 16)):
            raise RuntimeError(
                f"Wheel contains an unsafe member path: {info.filename!r}.")
        if _forbidden_path(info.filename, root):
            raise RuntimeError(
                f"Wheel contains a forbidden path: {info.filename}.")
        if info.filename.endswith(
            (".tar", ".gz", ".tgz", ".zip", ".whl", ".a", ".o", ".cubin",
             ".fatbin", ".ptx", ".map", ".bin")):
            raise RuntimeError(
                f"Wheel contains a build-only archive/kernel: {info.filename}."
            )
        with archive.open(info) as stream:
            audit_stream(stream, info.filename, root)


def require_policy(metadata: dict, root: Path = config.REPO_ROOT) -> None:
    """Reject legacy or differently sanitized wheel build inputs."""
    if enabled() and metadata.get("oss_policy_sha256") != policy_digest(root):
        raise RuntimeError(
            "Missing or stale OSS policy provenance; rebuild from sanitized sources."
        )
