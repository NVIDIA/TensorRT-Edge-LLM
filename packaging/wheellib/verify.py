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
"""Audit one native payload stage before wheel assembly."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Sequence, Set, Tuple

from .config import (CONTRACT, REPO_ROOT, cuda_driver_stub, load_matrix,
                     load_toml, require_variant, sha256)

_REQUIRED = set(CONTRACT.RUNTIME_VARIANT_FIELDS) | {
    "schema_version",
    "package_version",
    "cutedsl_groups",
    "evidence_sha256",
    "source_provenance_sha256",
    "submodule_revisions",
}


def _arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--allowlist",
                        type=Path,
                        default=REPO_ROOT / "packaging" /
                        "dependency_allowlist.toml")
    parser.add_argument("--size-budget",
                        type=Path,
                        default=REPO_ROOT / "packaging" / "size_budget.toml")
    parser.add_argument("--matrix",
                        type=Path,
                        default=REPO_ROOT / "packaging" / "variants.toml")
    parser.add_argument("--dependency-root",
                        action="append",
                        type=Path,
                        default=[],
                        help="Target sysroot or SDK library root.")
    parser.add_argument("--no-device-image-check",
                        action="store_true",
                        help=("Development-only: skip cuobjdump; "
                              "do not use for release artifacts."))
    return parser.parse_args(argv)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot read JSON metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON metadata {path} must be an object.")
    return value


def _stage_path(stage: Path, relative: str) -> Path:
    value = PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts:
        raise RuntimeError(
            f"Payload path must be relative and traversal-free: {relative!r}.")
    path = stage.joinpath(*value.parts)
    try:
        resolved_stage = stage.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"Payload path is missing: {path}.") from error
    if not resolved.is_relative_to(resolved_stage) or not resolved.is_file():
        raise RuntimeError(
            f"Payload path escapes its stage or is not a file: {relative!r}.")
    return resolved


def _audit_tool(name: str) -> str:
    executable = shutil.which(name)
    if executable is not None:
        return executable
    environment_tool = Path(sys.executable).parent / name
    if environment_tool.is_file() and os.access(environment_tool, os.X_OK):
        return os.fspath(environment_tool)
    raise RuntimeError(f"Required audit tool {name!r} is not installed.")


def _tool_output(argv: Iterable[str],
                 environment: Mapping[str, str] | None = None) -> str:
    command = list(argv)
    executable = _audit_tool(command[0])
    try:
        result = subprocess.run([executable, *command[1:]],
                                check=True,
                                text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                env=environment)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Audit command failed ({' '.join(command)}):\n{error.stdout}"
        ) from error
    return result.stdout


def _audit_elf(path: Path, cpu_arch: str, allowed: Set[str]) -> Set[str]:
    header = _tool_output(["readelf", "-h", os.fspath(path)])
    expected_machine = "Advanced Micro Devices X86-64" if cpu_arch == "x86_64" else "AArch64"
    if f"Machine:                           {expected_machine}" not in header:
        raise RuntimeError(
            f"ELF machine mismatch for {path}; expected {expected_machine}.")
    dynamic = _tool_output(["readelf", "-d", os.fspath(path)])
    needed = set(re.findall(r"\(NEEDED\).*?\[([^]]+)\]", dynamic))
    forbidden = sorted(needed - allowed)
    if forbidden:
        raise RuntimeError(
            f"{path} has non-allowlisted DT_NEEDED entries: {forbidden}.")
    for rpath in re.findall(r"\((?:RPATH|RUNPATH)\).*?\[([^]]+)\]", dynamic):
        for entry in rpath.split(":"):
            if entry and (entry not in {"$ORIGIN", "$ORIGIN/lib"}):
                raise RuntimeError(
                    f"{path} contains non-relative RPATH entry {entry!r}.")
    return needed


def _audit_device_images(path: Path, gpu_sm: int) -> None:
    executable = _audit_tool("cuobjdump")
    result = subprocess.run(
        [executable, "--dump-elf", os.fspath(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    pattern = rf"arch\s*=\s*sm[_-]?{gpu_sm}(?:a)?(?:\D|$)"
    if re.search(pattern, result.stdout, re.IGNORECASE):
        return
    if result.returncode != 0:
        raise RuntimeError(
            f"Audit command failed (cuobjdump --dump-elf {path}):\n"
            f"{result.stdout}")
    raise RuntimeError(f"{path} contains no listed SM{gpu_sm} device image.")


_MULTIARCH_TRIPLES = {
    "aarch64": "aarch64-linux-gnu",
    "x86_64": "x86_64-linux-gnu",
}
_CUDA_TARGETS = {
    "aarch64": ("aarch64-linux", "sbsa-linux"),
    "x86_64": ("x86_64-linux", ),
}


def _target_library_directories(root: Path, cpu_arch: str) -> Tuple[Path, ...]:
    """Return bounded target library directories below one SDK root."""
    try:
        multiarch = _MULTIARCH_TRIPLES[cpu_arch]
        cuda_targets = _CUDA_TARGETS[cpu_arch]
    except KeyError as error:
        raise RuntimeError(
            f"Unsupported dependency-root architecture {cpu_arch!r}."
        ) from error
    relative = (
        Path("."),
        Path("lib"),
        Path("lib64"),
        Path("lib") / multiarch,
        Path("usr/lib"),
        Path("usr/lib64"),
        Path("usr/lib") / multiarch,
        Path("usr/lib") / multiarch / "nvidia",
        Path("usr/lib") / multiarch / "nvidia/current",
        Path("usr/lib") / multiarch / "tegra",
        Path("usr/lib") / multiarch / "tegra-egl",
        Path("usr") / multiarch / "lib",
        Path("usr") / multiarch / "lib64",
        Path("usr/local/cuda/lib64"),
        Path("targets") / multiarch / "lib",
    )
    candidates = [root / value for value in relative]
    for cuda_target in cuda_targets:
        candidates.append(root / "usr/local/cuda/targets" / cuda_target /
                          "lib")
        candidates.append(root / "targets" / cuda_target / "lib")
    candidates.extend(root.glob("usr/local/cuda*/lib64"))
    for cuda_target in cuda_targets:
        candidates.extend(
            root.glob(f"usr/local/cuda*/targets/{cuda_target}/lib"))
    return tuple(dict.fromkeys(path for path in candidates if path.is_dir()))


def _audit_dependency_roots(needed: Set[str], roots: Sequence[Path],
                            cpu_arch: str) -> None:
    """Require every DT_NEEDED name in bounded target library directories."""
    directories = []
    for root in roots:
        resolved = root.resolve(strict=True)
        if not resolved.is_dir():
            raise RuntimeError(
                f"Dependency root is not a directory: {resolved}.")
        directories.extend(_target_library_directories(resolved, cpu_arch))
    directories = list(dict.fromkeys(directories))
    available = {
        name
        for name in needed
        if any((directory / name).is_file() or (directory / name).is_symlink()
               for directory in directories)
    }
    missing = sorted(needed - available)
    if missing:
        raise RuntimeError(
            "Target dependency roots do not provide DT_NEEDED entries: "
            f"{missing}; searched={[str(path) for path in directories]}.")


def _validate_payload_metadata(
        stage: Path,
        matrix_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload = _read_json(stage / "payload.json")
    missing = sorted(_REQUIRED - set(payload))
    extra = sorted(set(payload) - _REQUIRED)
    if missing or extra or payload.get("schema_version") != 1:
        raise RuntimeError(
            f"Invalid payload metadata; missing={missing}, extra={extra}, "
            f"schema={payload.get('schema_version')}.")
    if not re.fullmatch(r"[0-9a-f]{40}", str(payload["source_revision"])):
        raise RuntimeError(
            "payload source_revision must be a full lowercase Git SHA.")
    if not re.fullmatch(r"cp3\d{2}", str(payload["python_abi"])):
        raise RuntimeError("payload python_abi must be cp3XY.")

    matrix, rows = load_matrix(matrix_path.resolve())
    row = require_variant(rows, str(payload["variant_id"]))
    if payload["python_abi"] not in matrix["qualified_python_abis"]:
        raise RuntimeError(
            f"Payload uses unqualified ABI {payload['python_abi']}.")
    fields = (
        "platform_family",
        "platform_release",
        "platform_probe_source",
        "platform_probe_values",
        "cpu_arch",
        "cuda_runtime_soname",
        "tensorrt_runtime_soname",
        "gpu_sm",
    )
    drift = {
        field: (payload.get(field), row.get(field))
        for field in fields if payload.get(field) != row.get(field)
    }
    if drift:
        raise RuntimeError(
            f"Payload identity differs from variants.toml: {drift}.")
    return payload, row


def _validate_payload_hash(path: Path, payload: Mapping[str, Any],
                           key: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", str(payload[key])):
        raise RuntimeError(f"{key} must be a lowercase SHA-256 value.")
    actual = sha256(path)
    if actual != payload[key]:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: {actual} != {payload[key]}.")


def _payload_binaries(stage: Path,
                      payload: Mapping[str, Any]) -> Tuple[Path, Path, Path]:
    package_stage = stage / "tensorrt_edgellm"
    payload_prefix = f"_native/payloads/{payload['variant_id']}/"
    extension_relative = str(payload["extension"])
    plugin_relative = str(payload["plugin"])
    expected_extension = (
        f"_edgellm_runtime.cpython-{str(payload['python_abi'])[2:]}-"
        f"{payload['cpu_arch']}-linux-gnu.so")
    if (not extension_relative.startswith(payload_prefix)
            or "/" in extension_relative[len(payload_prefix):]
            or extension_relative[len(payload_prefix):] != expected_extension):
        raise RuntimeError(
            "Extension SOABI architecture does not match the selected "
            f"{payload['cpu_arch']} payload layout.")
    if plugin_relative != payload_prefix + "lib/libNvInfer_edgellm_plugin.so":
        raise RuntimeError(
            "Plugin path does not match the selected variant layout.")

    extension = _stage_path(package_stage, extension_relative)
    plugin = _stage_path(package_stage, plugin_relative)
    _validate_payload_hash(extension, payload, "extension_sha256")
    _validate_payload_hash(plugin, payload, "plugin_sha256")
    digits = str(payload["python_abi"])[2:]
    if f"cpython-{digits}" not in extension.name:
        raise RuntimeError(
            f"Extension {extension.name} does not match ABI {payload['python_abi']}."
        )
    return package_stage, extension, plugin


def _validate_cutedsl_metadata(stage: Path, payload: Mapping[str, Any],
                               row: Mapping[str, Any]) -> Dict[str, Any]:
    metadata_path = stage / "cutedsl-metadata.json"
    metadata = _read_json(metadata_path)
    if sha256(metadata_path) != payload["cutedsl_metadata_sha256"]:
        raise RuntimeError("CuTe metadata digest does not match payload.json.")
    groups = sorted(metadata.get("groups") or [])
    if groups != sorted(payload["cutedsl_groups"]):
        raise RuntimeError(
            "CuTe group inventory differs between metadata and payload.json.")
    expected = {
        "arch": row["cpu_arch"],
        "artifact_tag": row["cute_dsl_artifact_tag"],
        "gpu_arch": row["cute_dsl_artifact_tag"],
    }
    drift = {
        key: (metadata.get(key), value)
        for key, value in expected.items() if metadata.get(key) != value
    }
    expected_cuda_major = str(row["cuda_ctk_version"]).split(".", 1)[0]
    actual_cuda_major = str(metadata.get("cuda_version", "")).split(".", 1)[0]
    if actual_cuda_major != expected_cuda_major:
        drift["cuda_version"] = (metadata.get("cuda_version"),
                                 row["cuda_ctk_version"])
    if drift:
        raise RuntimeError(
            f"CuTe metadata differs from variants.toml: {drift}.")
    return metadata


def _load_evidence(stage: Path,
                   payload: Mapping[str, Any]) -> Tuple[Path, Dict[str, Any]]:
    evidence_dir = stage / "evidence"
    evidence_path = evidence_dir / "evidence.json"
    evidence = _read_json(evidence_path)
    if sha256(evidence_path) != payload["evidence_sha256"]:
        raise RuntimeError("Evidence digest does not match payload.json.")
    return evidence_dir, evidence


def _audit_stage_inventory(stage: Path, extension: Path, plugin: Path,
                           evidence_dir: Path) -> Path:
    license_file = extension.parent / "licenses" / "LICENSE"
    expected_files = {
        (stage / "payload.json").resolve(),
        (stage / "cutedsl-metadata.json").resolve(),
        extension.resolve(),
        plugin.resolve(),
        license_file.resolve(),
        (evidence_dir / "evidence.json").resolve(),
    }
    for candidate in stage.rglob("*"):
        if candidate.is_symlink():
            raise RuntimeError(
                f"Payload stage contains a forbidden symlink: {candidate}.")
        if candidate.is_file() and candidate.resolve() not in expected_files:
            raise RuntimeError(
                f"Payload stage contains an undeclared file: {candidate}.")
    return license_file


def _audit_cutedsl_evidence(metadata: Mapping[str, Any],
                            evidence: Mapping[str, Any]) -> None:
    variants = metadata.get("variants")
    members = evidence.get("archive_members")
    symbols = evidence.get("generated_symbols")
    linked = set(evidence.get("linked_symbols") or [])
    defined = set(evidence.get("defined_symbols") or [])
    unresolved = set(evidence.get("unresolved_symbols") or [])
    if (not isinstance(variants, list) or not variants
            or not isinstance(members, list) or not members
            or not isinstance(symbols, list) or not symbols):
        raise RuntimeError("CuTe evidence has an incomplete inventory.")
    member_text = "\n".join(str(value) for value in members)
    missing_variants = sorted(variant for variant in variants
                              if variant not in member_text)
    missing_symbols = sorted(set(symbols) - linked - defined)
    unresolved_generated = sorted(set(symbols) & unresolved)
    if missing_variants or missing_symbols or unresolved_generated:
        raise RuntimeError("CuTe link evidence is incomplete: "
                           f"variants={missing_variants[:20]}, "
                           f"symbols={missing_symbols[:20]}, "
                           f"unresolved={unresolved_generated[:20]}.")

    hashes = (
        evidence.get("header_sha256"),
        evidence.get("link_map_sha256"),
        evidence.get("debug_sha256"),
    )
    if any(not isinstance(value, dict) or not value for value in hashes):
        raise RuntimeError("CuTe evidence is missing source or debug hashes.")
    archive_hash = evidence.get("archive_sha256")
    if not isinstance(archive_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", archive_hash):
        raise RuntimeError("CuTe evidence has an invalid archive digest.")


def _audit_local_dependencies(binary: Path, allow_python_symbols: bool,
                              environment: Mapping[str, str] | None) -> None:
    dependency_tree = _tool_output(
        ["auditwheel", "lddtree", os.fspath(binary)], environment)
    lowered_tree = dependency_tree.lower()
    if ("not found" in lowered_tree
            or re.search(r'["\']path["\']\s*:\s*(?:null|none)', lowered_tree)):
        raise RuntimeError(
            f"Target dependency audit found unresolved libraries for {binary}:\n"
            f"{dependency_tree}")

    ldd_output = _tool_output(["ldd", "-r", os.fspath(binary)], environment)
    if "not found" in ldd_output:
        raise RuntimeError(
            f"Target dependency resolution failed for {binary}:\n{ldd_output}")
    unresolved = []
    for line in ldd_output.splitlines():
        match = re.search(r"undefined symbol:\s*([^\s]+)", line)
        if match is None:
            continue
        symbol = match.group(1)
        if allow_python_symbols and symbol.startswith(("Py", "_Py")):
            continue
        unresolved.append(symbol)
    if unresolved:
        raise RuntimeError(
            f"Unexpected unresolved symbols in {binary}: {sorted(set(unresolved))}."
        )


def _audit_binary_dependencies(payload: Mapping[str, Any], extension: Path,
                               plugin: Path, row: Mapping[str, Any],
                               allowlist_path: Path,
                               dependency_roots: Sequence[Path],
                               require_dependency_resolution: bool) -> None:
    allowlist = load_toml(allowlist_path)
    common = allowlist.get("common", {})
    allowed = set(common.get("allowed", []))
    allowed.update(allowlist.get(payload["cpu_arch"], {}).get("allowed", []))
    platform_provided = set(common.get("platform_provided", []))
    if not platform_provided.issubset(allowed):
        raise RuntimeError(
            "Platform-provided dependencies must also be allowlisted: "
            f"{sorted(platform_provided - allowed)}.")
    needed = _audit_elf(extension, str(payload["cpu_arch"]), allowed)
    needed.update(_audit_elf(plugin, str(payload["cpu_arch"]), allowed))
    required_dsos = {
        payload["cuda_runtime_soname"], payload["tensorrt_runtime_soname"]
    }
    if not required_dsos.issubset(needed):
        raise RuntimeError(
            f"Payload does not link its declared DSOs: {sorted(required_dsos - needed)}."
        )
    if not require_dependency_resolution:
        return
    host_machine = platform.machine().lower()
    host_arch = {
        "amd64": "x86_64",
        "arm64": "aarch64"
    }.get(host_machine, host_machine)
    if host_arch != payload["cpu_arch"]:
        if dependency_roots:
            _audit_dependency_roots(needed - platform_provided,
                                    dependency_roots, str(payload["cpu_arch"]))
    else:
        environment = None
        with tempfile.TemporaryDirectory(
                prefix="edgellm-driver-stub-") as temporary:
            driver_stub = cuda_driver_stub(row)
            if ("libcuda.so.1" in platform_provided
                    and driver_stub is not None):
                Path(temporary, "libcuda.so.1").symlink_to(driver_stub)
                environment = dict(os.environ)
                environment["LD_LIBRARY_PATH"] = ":".join(
                    value for value in (temporary,
                                        environment.get("LD_LIBRARY_PATH", ""))
                    if value)
            _audit_local_dependencies(extension, True, environment)
            _audit_local_dependencies(plugin, False, environment)


def _audit_payload_size(package_stage: Path, size_budget_path: Path,
                        cpu_arch: str) -> None:
    budget = int(
        load_toml(size_budget_path)["payload"][cpu_arch]["unpacked_bytes"])
    unpacked = sum(path.stat().st_size for path in package_stage.rglob("*")
                   if path.is_file())
    if unpacked > budget:
        raise RuntimeError(
            f"Payload size {unpacked} exceeds reviewed budget {budget}.")


def verify(stage: Path,
           allowlist_path: Path,
           size_budget_path: Path,
           *,
           matrix_path: Path = REPO_ROOT / "packaging" / "variants.toml",
           require_device_images: bool = True,
           dependency_roots: Sequence[Path] = (),
           require_dependency_resolution: bool = True) -> Dict[str, Any]:
    """Validate metadata, content, dependencies, architecture, and size."""
    stage = stage.resolve(strict=True)
    payload, row = _validate_payload_metadata(stage, matrix_path)
    package_stage, extension, plugin = _payload_binaries(stage, payload)
    cutedsl_metadata = _validate_cutedsl_metadata(stage, payload, row)
    evidence_dir, evidence = _load_evidence(stage, payload)
    license_file = _audit_stage_inventory(stage, extension, plugin,
                                          evidence_dir)
    if (not license_file.is_file()
            or sha256(license_file) != sha256(REPO_ROOT / "LICENSE")):
        raise RuntimeError(
            "Payload is missing the canonical, unmodified LICENSE file.")
    _audit_cutedsl_evidence(cutedsl_metadata, evidence)
    _audit_binary_dependencies(payload, extension, plugin, row, allowlist_path,
                               dependency_roots, require_dependency_resolution)
    if require_device_images:
        _audit_device_images(extension, int(payload["gpu_sm"]))
        _audit_device_images(plugin, int(payload["gpu_sm"]))
    _audit_payload_size(package_stage, size_budget_path,
                        str(payload["cpu_arch"]))
    return payload


def main(argv=None) -> None:
    """Run the command-line payload audit."""
    args = _arguments(argv)
    payload = verify(args.stage,
                     args.allowlist,
                     args.size_budget,
                     matrix_path=args.matrix,
                     require_device_images=not args.no_device_image_check,
                     dependency_roots=args.dependency_root)
    print(f"verified {payload['variant_id']} {payload['python_abi']}")


if __name__ == "__main__":
    main()
