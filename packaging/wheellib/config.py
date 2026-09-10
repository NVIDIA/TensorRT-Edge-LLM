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
"""Shared standard-library helpers for EdgeLLM wheel packaging."""

from __future__ import annotations

import configparser
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Set, Tuple)

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 build hosts.
    try:
        import tomli as tomllib
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Python 3.10 packaging hosts require tomli: python -m pip install tomli"
        ) from error


def repository_root(start: Optional[Path] = None) -> Path:
    """Return the Git worktree root containing *start*."""
    location = (start or Path(__file__).resolve().parent).resolve()
    try:
        result = subprocess.run(
            ["git", "-C",
             str(location), "rev-parse", "--show-toplevel"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"Cannot determine the Git repository root from {location}."
        ) from error
    root = Path(result.stdout.strip()).resolve()
    if not root.is_dir():
        raise RuntimeError(f"Git returned an invalid repository root: {root}.")
    return root


REPO_ROOT = repository_root()


def _load_contract():
    path = REPO_ROOT / "tensorrt_edgellm" / "_native" / "contract.py"
    spec = importlib.util.spec_from_file_location(
        "_tensorrt_edgellm_wheel_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the wheel contract from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTRACT = _load_contract()


def required_environment(name: str) -> str:
    """Return one required, non-empty environment variable."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is unset.")
    return value


def cuda_driver_stub(row: Mapping[str, Any]) -> Optional[Path]:
    """Return the target CUDA driver stub when the toolkit provides one."""
    cuda_dir = Path(
        os.environ.get("CUDA_DIR",
                       f"/usr/local/cuda-{row['cuda_ctk_version']}"))
    target_arch = {
        "x86_64": "x86_64-linux",
        "aarch64": "aarch64-linux",
    }[str(row["cpu_arch"])]
    candidates = (
        cuda_dir / "lib64" / "stubs" / "libcuda.so",
        cuda_dir / "lib" / "stubs" / "libcuda.so",
        cuda_dir / "targets" / target_arch / "lib" / "stubs" / "libcuda.so",
    )
    return next((path.resolve() for path in candidates if path.is_file()),
                None)


def load_toml(path: Path) -> Dict[str, Any]:
    """Load TOML from *path* and return its top-level mapping."""
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(
            f"Cannot read TOML policy {path}: {error}") from error
    if not isinstance(data, dict):
        raise RuntimeError(
            f"TOML policy {path} must contain a top-level table.")
    return data


def sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checked(argv: Sequence[str],
                *,
                cwd: Optional[Path] = None,
                env: Optional[Mapping[str, str]] = None) -> None:
    """Run a required command and raise an actionable error on failure."""
    executable = shutil.which(argv[0])
    if executable is None:
        raise RuntimeError(
            f"Required executable {argv[0]!r} is not installed.")
    command = [executable, *argv[1:]]
    try:
        subprocess.run(command, cwd=cwd, env=env, check=True)
    except subprocess.CalledProcessError as error:
        rendered = " ".join(command)
        raise RuntimeError(
            f"Command failed with exit code {error.returncode}: {rendered}"
        ) from error


def package_version(repo_root: Path = REPO_ROOT) -> str:
    """Read the package version without importing its dependency-heavy package."""
    version_file = repo_root / "tensorrt_edgellm" / "_version.py"
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']',
                      version_file.read_text(encoding="utf-8"), re.MULTILINE)
    if not match:
        raise RuntimeError(
            f"Cannot determine package version from {version_file}.")
    return match.group(1)


def source_revision(repo_root: Path = REPO_ROOT) -> str:
    """Return the full source Git revision."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"],
                                cwd=repo_root,
                                check=True,
                                text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Cannot determine the source Git revision.") from error
    revision = result.stdout.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError(f"Unexpected Git revision {revision!r}.")
    return revision


def require_clean_source(repo_root: Path = REPO_ROOT) -> None:
    """Reject tracked source modifications in a wheel build checkout."""
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE)
    if result.stdout.strip():
        raise RuntimeError(
            "Wheel packaging requires a clean tracked checkout; commit or revert changes, "
            "or use the explicit development override.")


def _parse_submodule_status(output: str) -> Dict[str, str]:
    """Parse clean recursive ``git submodule status`` output."""
    revisions = {}
    for line in output.splitlines():
        if not line:
            continue
        match = re.match(r"^(.)([0-9a-f]{40})\s+(\S+)", line)
        if match is None or match.group(1) != " ":
            raise RuntimeError(
                f"Uninitialized or out-of-sync submodule state: {line}")
        revisions[match.group(3)] = match.group(2)
    return revisions


def _submodule_revisions(repo_root: Path) -> Dict[str, str]:
    result = subprocess.run(["git", "submodule", "status", "--recursive"],
                            cwd=repo_root,
                            check=True,
                            text=True,
                            stdout=subprocess.PIPE)
    return _parse_submodule_status(result.stdout)


def _declared_submodule_revisions(repo_root: Path) -> Dict[str, str]:
    config_path = repo_root / ".gitmodules"
    parser = configparser.ConfigParser(interpolation=None)
    try:
        with config_path.open(encoding="utf-8") as stream:
            parser.read_file(stream)
    except (OSError, configparser.Error) as error:
        raise RuntimeError(
            f"Cannot read submodule revisions from {config_path}.") from error

    revisions = {}
    for section in parser.sections():
        if not section.startswith('submodule "'):
            continue
        values = parser[section]
        if "path" not in values or "revision" not in values:
            raise RuntimeError(
                f"{config_path} section {section!r} must declare path and revision."
            )
        submodule_path = values["path"]
        revision = values["revision"].lower()
        relative = Path(submodule_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(
                f"Unsafe submodule path in {config_path}: {submodule_path!r}.")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise RuntimeError(
                f"Submodule {submodule_path} must declare an exact Git SHA.")
        if submodule_path in revisions:
            raise RuntimeError(
                f"Duplicate submodule path in {config_path}: {submodule_path}."
            )
        revisions[submodule_path] = revision
    if not revisions:
        raise RuntimeError(f"{config_path} declares no submodule revisions.")
    return revisions


def _gitlink_revisions(repo_root: Path,
                       paths: Iterable[str]) -> Dict[str, str]:
    revisions = {}
    for path in paths:
        result = subprocess.run(["git", "ls-tree", "HEAD", "--", path],
                                cwd=repo_root,
                                check=True,
                                text=True,
                                stdout=subprocess.PIPE)
        fields = result.stdout.strip().split()
        if len(fields) < 4 or fields[0] != "160000":
            raise RuntimeError(f"Repository has no gitlink for {path}.")
        revisions[path] = fields[2].lower()
    return revisions


def _validate_submodules(repo_root: Path) -> Dict[str, str]:
    declared = _declared_submodule_revisions(repo_root)
    gitlinks = _gitlink_revisions(repo_root, declared)
    checked_out = _submodule_revisions(repo_root)
    if set(declared) != set(checked_out):
        raise RuntimeError(
            "Submodule declaration/status drift: "
            f"missing={sorted(set(declared) - set(checked_out))}, "
            f"undeclared={sorted(set(checked_out) - set(declared))}.")
    mismatches = {
        path: {
            "declared": declared[path],
            "gitlink": gitlinks[path],
            "checkout": checked_out[path],
        }
        for path in declared
        if len({declared[path], gitlinks[path], checked_out[path]}) != 1
    }
    if mismatches:
        raise RuntimeError(
            f"Declared, gitlink, and checked-out submodule SHAs differ: {mismatches}."
        )
    dirty = []
    for path in declared:
        result = subprocess.run(
            ["git", "-C",
             str(repo_root / path), "status", "--porcelain"],
            check=True,
            text=True,
            stdout=subprocess.PIPE)
        if result.stdout.strip():
            dirty.append(path)
    if dirty:
        raise RuntimeError(
            f"Dirty submodule checkouts are unsupported: {dirty}.")
    return checked_out


def source_snapshot(repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    """Return compact source provenance bound to Git, matrix, and submodules."""
    submodules = _validate_submodules(repo_root)
    matrix_path = repo_root / "packaging" / "variants.toml"
    materials = {
        "source_revision": source_revision(repo_root),
        "matrix_sha256": sha256(matrix_path),
        "submodule_revisions": submodules,
    }
    serialized = json.dumps(materials, sort_keys=True,
                            separators=(",", ":")).encode()
    return {
        **materials,
        "sha256": hashlib.sha256(serialized).hexdigest(),
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write deterministic UTF-8 JSON, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _cutedsl_inventory_from_readme(
        repo_root: Path) -> Set[Tuple[str, int, int]]:
    readme = (repo_root / "kernelSrcs" /
              "README.md").read_text(encoding="utf-8")
    matches = re.findall(
        r"cutedsl_(x86_64|aarch64)_sm_(\d+)_cuda(\d+)\.tar\.gz", readme)
    return {(arch, int(sm), int(cuda)) for arch, sm, cuda in matches}


def _validate_matrix_header(
        matrix: Mapping[str, Any]) -> Tuple[List[str], Set[int]]:
    if matrix.get("schema_version") != 1:
        raise RuntimeError("variants.toml must use schema_version = 1.")
    python_abis = matrix.get("qualified_python_abis")
    if not isinstance(python_abis, list) or not python_abis:
        raise RuntimeError("variants.toml must declare qualified_python_abis.")
    if len(set(python_abis)) != len(python_abis) or any(
            not re.fullmatch(r"cp3\d{2}", value) for value in python_abis):
        raise RuntimeError(
            "qualified_python_abis must be unique cp3XY values.")
    trt_majors = set(matrix.get("qualified_tensorrt_majors", []))
    if not trt_majors or any(not isinstance(value, int)
                             for value in trt_majors):
        raise RuntimeError(
            "variants.toml must declare integer qualified_tensorrt_majors.")
    return python_abis, trt_majors


def _validate_cutedsl_artifacts(matrix: Mapping[str, Any],
                                repo_root: Path) -> Set[Tuple[str, int, int]]:
    artifacts = matrix.get("cutedsl_artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("variants.toml must declare cutedsl_artifacts.")
    try:
        declared = {(str(entry["cpu_arch"]), int(entry["gpu_sm"]),
                     int(entry["cuda_major"]))
                    for entry in artifacts}
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "variants.toml has an invalid CuTe artifact row.") from error

    documented_artifacts = _cutedsl_inventory_from_readme(repo_root)
    if declared != documented_artifacts:
        missing = sorted(documented_artifacts - declared)
        invented = sorted(declared - documented_artifacts)
        raise RuntimeError(
            "CuTe artifact inventory drift: "
            f"missing documented pairs={missing}, invented pairs={invented}.")
    return declared


_EMBEDDED_TARGETS = {"", "auto-thor", "gb10", "jetson-orin", "jetson-thor"}


def _normalize_variant(value: Any, index: int) -> Dict[str, Any]:
    try:
        return CONTRACT.validate_matrix_variant(value)
    except ValueError as error:
        raise RuntimeError(f"Invalid variant row {index}: {error}") from error


def _validate_variant_platform(row: Mapping[str, Any]) -> None:
    variant_id = str(row["variant_id"])
    embedded_target = row["embedded_target"]
    if embedded_target not in _EMBEDDED_TARGETS:
        raise RuntimeError(
            f"Variant {variant_id} uses unsupported EMBEDDED_TARGET "
            f"{embedded_target!r}.")
    if row["cpu_arch"] == "x86_64" and embedded_target:
        raise RuntimeError(
            f"x86 variant {variant_id} cannot set EMBEDDED_TARGET.")
    if row["cpu_arch"] == "aarch64" and not embedded_target:
        raise RuntimeError(
            f"aarch64 variant {variant_id} must set EMBEDDED_TARGET.")


def _validate_variant_dependencies(row: Mapping[str, Any],
                                   declared_artifacts: Set[Tuple[str, int,
                                                                 int]],
                                   trt_majors: Set[int]) -> None:
    variant_id = str(row["variant_id"])
    try:
        cuda_major = int(str(row["cuda_ctk_version"]).split(".", 1)[0])
        gpu_sm = int(row["gpu_sm"])
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Variant {variant_id} has invalid CUDA or SM metadata."
        ) from error
    artifact = (str(row["cpu_arch"]), gpu_sm, cuda_major)
    if artifact not in declared_artifacts:
        raise RuntimeError(
            f"Variant {variant_id} references undeclared CuTe artifact {artifact}."
        )
    if row["cute_dsl_artifact_tag"] != f"sm_{gpu_sm}":
        raise RuntimeError(
            f"Variant {variant_id} artifact tag does not match gpu_sm.")
    cudart_match = re.fullmatch(r"libcudart\.so\.(\d+)",
                                str(row["cuda_runtime_soname"]))
    if cudart_match is None or int(cudart_match.group(1)) != cuda_major:
        raise RuntimeError(
            f"Variant {variant_id} CUDA SONAME does not match cuda_ctk_version."
        )
    trt_match = re.fullmatch(r"libnvinfer\.so\.(\d+)",
                             str(row["tensorrt_runtime_soname"]))
    if trt_match is None or int(trt_match.group(1)) not in trt_majors:
        raise RuntimeError(
            f"Variant {variant_id} uses an unqualified TensorRT runtime major."
        )


def _runtime_identity(row: Mapping[str, Any]) -> Tuple[object, ...]:
    return (
        row["platform_family"],
        row["platform_release"],
        row["cpu_arch"],
        row["cuda_runtime_soname"],
        row["tensorrt_runtime_soname"],
        int(row["gpu_sm"]),
    )


def _validate_required_releases(matrix: Mapping[str, Any],
                                releases: Set[str]) -> None:
    support = matrix.get("support", {})
    if not isinstance(support, dict):
        raise RuntimeError("variants.toml support must be a table.")
    required = set(support.get("required_platform_releases", []))
    if releases != required:
        raise RuntimeError("Supported platform release drift: "
                           f"missing={sorted(required - releases)}, "
                           f"invented={sorted(releases - required)}.")


def _validate_variant_rows(
        values: Sequence[Any], declared_artifacts: Set[Tuple[str, int, int]],
        trt_majors: Set[int]) -> Tuple[List[Dict[str, Any]], Set[str]]:
    ids: Set[str] = set()
    identities: Set[Tuple[object, ...]] = set()
    releases: Set[str] = set()
    normalized = []
    for index, value in enumerate(values):
        row = _normalize_variant(value, index)
        variant_id = str(row["variant_id"])
        if variant_id in ids:
            raise RuntimeError(f"Duplicate variant_id {variant_id!r}.")
        ids.add(variant_id)
        _validate_variant_platform(row)
        _validate_variant_dependencies(row, declared_artifacts, trt_majors)
        identity = _runtime_identity(row)
        if identity in identities:
            raise RuntimeError(
                f"Duplicate runtime identity in variant {variant_id}.")
        identities.add(identity)
        releases.add(f"{row['platform_family']}:{row['platform_release']}")
        normalized.append(row)
    return normalized, releases


def _matrix_variants(matrix: Mapping[str, Any]) -> Sequence[Any]:
    variants = matrix.get("variants")
    if not isinstance(variants, list) or not variants:
        raise RuntimeError(
            "variants.toml must declare at least one [[variants]] row.")
    return variants


def validate_matrix(matrix: Mapping[str, Any],
                    repo_root: Path = REPO_ROOT) -> List[Dict[str, Any]]:
    """Validate qualification rows against support and CuTe inventory declarations."""
    _, trt_majors = _validate_matrix_header(matrix)
    declared_artifacts = _validate_cutedsl_artifacts(matrix, repo_root)
    normalized, releases = _validate_variant_rows(_matrix_variants(matrix),
                                                  declared_artifacts,
                                                  trt_majors)
    _validate_required_releases(matrix, releases)
    return normalized


def load_matrix(
    path: Path,
    repo_root: Path = REPO_ROOT
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load and validate the wheel qualification matrix."""
    matrix = load_toml(path)
    return matrix, validate_matrix(matrix, repo_root)


def require_variant(rows: Iterable[Mapping[str, Any]],
                    variant_id: str) -> Dict[str, Any]:
    """Return one named matrix row or raise with available names."""
    matches = [
        dict(row) for row in rows if row.get("variant_id") == variant_id
    ]
    if len(matches) != 1:
        names = ", ".join(sorted(str(row.get("variant_id")) for row in rows))
        raise RuntimeError(
            f"Unknown or duplicate variant {variant_id!r}; available variants: {names}."
        )
    return matches[0]
