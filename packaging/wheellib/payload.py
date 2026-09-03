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
"""Build one exact target and CPython native payload stage."""

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
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from .config import (REPO_ROOT, cuda_driver_stub, load_matrix, package_version,
                     require_clean_source, require_variant, run_checked,
                     sha256, source_revision, source_snapshot, write_json)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant",
                        required=True,
                        help="variant_id from variants.toml.")
    parser.add_argument("--python-abi",
                        required=True,
                        help="Qualified ABI such as cp310.")
    parser.add_argument("--trt-package-dir",
                        type=Path,
                        required=True,
                        help="Target TensorRT package root.")
    parser.add_argument("--output-dir",
                        type=Path,
                        required=True,
                        help="Fresh payload stage directory.")
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="CMake build directory (default: build/wheel/<row>-<abi>).")
    parser.add_argument("--build-jobs",
                        type=_positive_int,
                        help="Maximum parallel CMake build processes.")
    parser.add_argument(
        "--compiler-launcher",
        type=Path,
        help="Optional compiler launcher shared by C, C++, and CUDA builds.")
    parser.add_argument("--matrix",
                        type=Path,
                        default=REPO_ROOT / "packaging" / "variants.toml")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-revision",
                        help="Expected full Git revision.")
    parser.add_argument("--toolchain-file",
                        type=Path,
                        help="CMake toolchain override for cross compilation.")
    parser.add_argument(
        "--target-sysroot",
        type=Path,
        help="Target SDK sysroot required for cross compilation.")
    parser.add_argument(
        "--target-python-include-dir",
        type=Path,
        help="Target CPython headers required for a cross build.")
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="Development-only override; do not use for release artifacts.")
    return parser.parse_args(argv)


def _current_python_abi() -> str:
    tag = getattr(sys.implementation, "cache_tag", "") or ""
    match = re.fullmatch(r"cpython-(\d{2,3})", tag)
    if not match:
        raise RuntimeError(
            f"Payload builds require GIL CPython; detected {tag!r}.")
    return f"cp{match.group(1)}"


def _one_file(paths, label: str) -> Path:
    values = list(paths)
    if len(values) != 1:
        raise RuntimeError(
            f"Expected one {label}, found {len(values)}: {values}.")
    return values[0]


def _cmake_tool(build_dir: Path, name: str) -> str:
    """Return a target-aware binutils executable from CMakeCache.txt."""
    cache = (build_dir / "CMakeCache.txt").read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(name)}:FILEPATH=(.+)$", cache,
                      re.MULTILINE)
    if not match or not Path(match.group(1)).is_file():
        raise RuntimeError(
            f"CMake did not resolve required target tool {name}.")
    return match.group(1)


def _host_arch() -> str:
    machine = platform.machine().lower()
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)


def _validated_inputs(
    args: argparse.Namespace
) -> Tuple[Path, Dict[str, Any], Path, str, Dict[str, Any]]:
    repo_root = args.repo_root.resolve()
    matrix, rows = load_matrix(args.matrix.resolve(), repo_root)
    row = require_variant(rows, args.variant)
    if args.python_abi not in matrix["qualified_python_abis"]:
        raise RuntimeError(
            f"Python ABI {args.python_abi!r} is not qualified by variants.toml."
        )
    current_abi = _current_python_abi()
    if args.python_abi != current_abi:
        raise RuntimeError(
            f"Build interpreter ABI {current_abi} does not match requested {args.python_abi}."
        )
    trt_package_dir = args.trt_package_dir.resolve()
    if not (trt_package_dir / "include" / "NvInfer.h").is_file():
        raise RuntimeError(
            f"TensorRT headers were not found under {trt_package_dir}/include."
        )
    checkout_revision = source_revision(repo_root)
    revision = args.source_revision or checkout_revision
    if revision != checkout_revision:
        raise RuntimeError(
            f"Requested revision {revision} does not match checkout HEAD.")
    if not args.allow_dirty_source:
        require_clean_source(repo_root)
    return repo_root, row, trt_package_dir, revision, source_snapshot(
        repo_root)


def _build_directories(args: argparse.Namespace,
                       repo_root: Path) -> Tuple[Path, Path]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Payload output directory must be empty: {output_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir = (args.build_dir or repo_root / "build" / "wheel" /
                 f"{args.variant}-{args.python_abi}").resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, build_dir


def _cross_cmake_options(args: argparse.Namespace,
                         row: Mapping[str, Any]) -> List[str]:
    host_arch = _host_arch()
    cross_build = host_arch != row["cpu_arch"]
    if cross_build and args.toolchain_file is None:
        raise RuntimeError(
            f"Cross-building {row['cpu_arch']} from {host_arch} requires --toolchain-file."
        )
    if cross_build and args.target_sysroot is None:
        raise RuntimeError(
            f"Cross-building {row['cpu_arch']} from {host_arch} requires --target-sysroot."
        )
    if cross_build and args.target_python_include_dir is None:
        raise RuntimeError(
            "Cross-building the pybind11 extension requires --target-python-include-dir "
            "for the requested CPython minor.")

    options = []
    if args.toolchain_file is not None:
        options.append(
            f"-DCMAKE_TOOLCHAIN_FILE={args.toolchain_file.resolve(strict=True)}"
        )
    if args.target_sysroot is not None:
        options.append(
            f"-DCMAKE_SYSROOT={args.target_sysroot.resolve(strict=True)}")
    if args.target_python_include_dir is not None:
        include_dir = args.target_python_include_dir.resolve(strict=True)
        if not (include_dir / "Python.h").is_file():
            raise RuntimeError(
                f"Target Python.h was not found under {include_dir}.")
        options.append(f"-DPython_INCLUDE_DIR={include_dir}")
    return options


def _pybind11_cmake_dir() -> Path:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pybind11", "--cmakedir"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "pybind11 must be installed in the payload-build environment."
        ) from error
    cmake_dir = Path(result.stdout.strip()).resolve()
    if not (cmake_dir / "pybind11Config.cmake").is_file():
        raise RuntimeError(
            f"pybind11 did not report a usable CMake package: {cmake_dir}.")
    return cmake_dir


def _cuda_architecture(row: Mapping[str, Any]) -> str:
    sm = int(row["gpu_sm"])
    suffix = "a" if sm in {100, 101, 110, 121} else ""
    return f"{sm}{suffix}"


def _cmake_configure_command(args: argparse.Namespace, repo_root: Path,
                             build_dir: Path, trt_package_dir: Path,
                             row: Mapping[str, Any],
                             payload_rel: Path) -> List[str]:
    abi_digits = args.python_abi[2:]
    cuda_architecture = _cuda_architecture(row)
    extension_name = (
        f"_edgellm_runtime.cpython-{abi_digits}-{row['cpu_arch']}-linux-gnu.so"
    )
    command = [
        "cmake",
        "-S",
        os.fspath(repo_root),
        "-B",
        os.fspath(build_dir),
        "-UPython_*",
        "-U_Python_*",
        "-UPYTHON_*",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architecture}",
        "-DBUILD_PYTHON_BINDINGS=ON",
        f"-DTRT_PACKAGE_DIR={trt_package_dir}",
        f"-DCUDA_CTK_VERSION={row['cuda_ctk_version']}",
        f"-DCUTE_DSL_ARTIFACT_TAG={row['cute_dsl_artifact_tag']}",
        f"-DEMBEDDED_TARGET={row['embedded_target']}",
        f"-DEDGELLM_WHEEL_PAYLOAD_DIR={payload_rel.as_posix()}",
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-Dpybind11_DIR={_pybind11_cmake_dir()}",
        f"-DEDGELLM_WHEEL_EXTENSION_NAME={extension_name}",
    ]
    driver_stub = cuda_driver_stub(row)
    if driver_stub is not None:
        command.append(f"-DCUDA_DRIVER_LIB={driver_stub}")
    if args.compiler_launcher is not None:
        launcher = args.compiler_launcher.resolve(strict=True)
        if not os.access(launcher, os.X_OK):
            raise RuntimeError(
                f"Compiler launcher is not executable: {launcher}.")
        command.extend(f"-DCMAKE_{language}_COMPILER_LAUNCHER={launcher}"
                       for language in ("C", "CXX", "CUDA"))
    command.extend(_cross_cmake_options(args, row))
    command.extend(str(value) for value in row["cmake_args"])
    return command


def _build_and_install(args: argparse.Namespace, repo_root: Path,
                       build_dir: Path, output_dir: Path,
                       trt_package_dir: Path, row: Mapping[str, Any],
                       payload_rel: Path) -> None:
    run_checked(
        _cmake_configure_command(args, repo_root, build_dir, trt_package_dir,
                                 row, payload_rel))
    build_command = ["cmake", "--build", os.fspath(build_dir), "--parallel"]
    if args.build_jobs is not None:
        build_command.append(str(args.build_jobs))
    build_command.extend(
        ["--target", "_edgellm_runtime", "NvInfer_edgellm_plugin"])
    run_checked(build_command)
    run_checked([
        "cmake", "--install",
        os.fspath(build_dir), "--prefix",
        os.fspath(output_dir)
    ])


def _installed_binaries(output_dir: Path,
                        payload_rel: Path) -> Tuple[Path, Path]:
    payload_dir = output_dir / payload_rel
    extension = _one_file(payload_dir.glob("_edgellm_runtime*.so"),
                          "pybind11 extension")
    plugin = _one_file(
        (payload_dir / "lib").glob("libNvInfer_edgellm_plugin.so"),
        "EdgeLLM plugin")
    return extension, plugin


def _extract_debug_symbols(build_dir: Path, output_dir: Path,
                           binaries: Tuple[Path, Path]) -> Path:
    debug_dir = output_dir / "evidence" / "debug"
    debug_dir.mkdir(parents=True)
    objcopy = _cmake_tool(build_dir, "CMAKE_OBJCOPY")
    strip = _cmake_tool(build_dir, "CMAKE_STRIP")
    for binary in binaries:
        debug_file = debug_dir / f"{binary.name}.debug"
        run_checked([
            objcopy, "--only-keep-debug",
            os.fspath(binary),
            os.fspath(debug_file)
        ])
        run_checked([strip, "--strip-unneeded", os.fspath(binary)])
        run_checked(
            [objcopy, f"--add-gnu-debuglink={debug_file}",
             os.fspath(binary)])
    return debug_dir


def _cutedsl_inputs(
    repo_root: Path,
    row: Mapping[str,
                 Any]) -> Tuple[Path, Dict[str, Any], List[str], Path, Path]:
    metadata_path = (repo_root / "cpp" / "kernels" / "cuteDSLArtifact" /
                     str(row["cpu_arch"]) / str(row["cute_dsl_artifact_tag"]) /
                     "metadata.json")
    if not metadata_path.is_file():
        raise RuntimeError(
            f"CuTe metadata was not produced or extracted: {metadata_path}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    groups = sorted(metadata.get("groups") or [])
    if not groups:
        raise RuntimeError(f"CuTe metadata has no groups: {metadata_path}.")
    artifact_dir = metadata_path.parent
    archive = artifact_dir / f"libcutedsl_{row['cpu_arch']}.a"
    headers = artifact_dir / "include"
    if not archive.is_file() or not headers.is_dir():
        raise RuntimeError(f"Incomplete CuTe artifact under {artifact_dir}.")
    return metadata_path, metadata, groups, archive, headers


def _archive_member_inventory(archive: Path) -> str:
    try:
        members = subprocess.run(["ar", "t", os.fspath(archive)],
                                 check=True,
                                 text=True,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"Cannot inventory CuTe archive {archive}.") from error
    if not members.strip():
        raise RuntimeError(f"CuTe archive is empty: {archive}.")
    return members


def _nm_symbols(path: Path, option: str) -> str:
    try:
        return subprocess.run(
            ["nm", option, os.fspath(path)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Cannot inspect symbols in {path}.") from error


def _collect_evidence(output_dir: Path, build_dir: Path, archive: Path,
                      headers: Path, debug_dir: Path) -> Path:
    link_map_dir = build_dir / "wheel-link-maps"
    maps = [link_map_dir / "extension.map", link_map_dir / "plugin.map"]
    if any(not path.is_file() for path in maps):
        raise RuntimeError(
            f"Native link maps were not generated under {link_map_dir}.")

    header_files = sorted(path for path in headers.rglob("*")
                          if path.is_file())
    header_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in header_files)
    symbols = sorted(
        set(
            re.findall(
                r"^\s*void\s+(_mlir_[A-Za-z_]\w*)\s*\([^;{}]*\)\s*;\s*$",
                header_text,
                re.MULTILINE,
            )))
    link_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in maps)
    debug_files = sorted(debug_dir.iterdir())
    defined = "".join(
        _nm_symbols(path, "--defined-only") for path in debug_files)
    unresolved = "".join(
        _nm_symbols(path, "--undefined-only") for path in debug_files)

    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    evidence = {
        "archive_sha256":
        sha256(archive),
        "archive_members":
        _archive_member_inventory(archive).splitlines(),
        "header_sha256": {
            path.relative_to(headers).as_posix(): sha256(path)
            for path in header_files
        },
        "generated_symbols":
        symbols,
        "linked_symbols":
        [symbol for symbol in symbols if symbol in link_text],
        "defined_symbols": [symbol for symbol in symbols if symbol in defined],
        "unresolved_symbols":
        [symbol for symbol in symbols if symbol in unresolved],
        "link_map_sha256": {
            path.name: sha256(path)
            for path in maps
        },
        "debug_sha256": {
            path.name: sha256(path)
            for path in debug_files
        },
    }
    write_json(evidence_dir / "evidence.json", evidence)
    return evidence_dir


def _payload_manifest(args: argparse.Namespace, repo_root: Path,
                      output_dir: Path, row: Mapping[str, Any], revision: str,
                      snapshot: Mapping[str, Any], extension: Path,
                      plugin: Path, cutedsl_metadata: Path, groups: List[str],
                      evidence_dir: Path) -> Dict[str, Any]:
    return {
        "schema_version":
        1,
        "package_version":
        package_version(repo_root),
        "source_revision":
        revision,
        "variant_id":
        args.variant,
        "python_abi":
        args.python_abi,
        "platform_family":
        row["platform_family"],
        "platform_release":
        row["platform_release"],
        "platform_probe_source":
        row["platform_probe_source"],
        "platform_probe_values":
        row["platform_probe_values"],
        "cpu_arch":
        row["cpu_arch"],
        "cuda_runtime_soname":
        row["cuda_runtime_soname"],
        "tensorrt_runtime_soname":
        row["tensorrt_runtime_soname"],
        "gpu_sm":
        int(row["gpu_sm"]),
        "extension":
        extension.relative_to(output_dir / "tensorrt_edgellm").as_posix(),
        "plugin":
        plugin.relative_to(output_dir / "tensorrt_edgellm").as_posix(),
        "extension_sha256":
        sha256(extension),
        "plugin_sha256":
        sha256(plugin),
        "cutedsl_metadata_sha256":
        sha256(cutedsl_metadata),
        "cutedsl_groups":
        groups,
        "evidence_sha256":
        sha256(evidence_dir / "evidence.json"),
        "source_provenance_sha256":
        snapshot["sha256"],
        "submodule_revisions":
        snapshot["submodule_revisions"],
    }


def main(argv=None) -> None:
    """Configure, build, install, and describe one isolated native payload."""
    args = _arguments(argv)
    repo_root, row, trt_dir, revision, snapshot = _validated_inputs(args)
    output_dir, build_dir = _build_directories(args, repo_root)
    payload_rel = (Path("tensorrt_edgellm") / "_native" / "payloads" /
                   args.variant)
    _build_and_install(args, repo_root, build_dir, output_dir, trt_dir, row,
                       payload_rel)
    extension, plugin = _installed_binaries(output_dir, payload_rel)
    metadata_path, _, groups, archive, headers = _cutedsl_inputs(
        repo_root, row)
    with tempfile.TemporaryDirectory(prefix="edgellm-debug-") as temporary:
        debug_dir = _extract_debug_symbols(build_dir, Path(temporary),
                                           (extension, plugin))
        evidence_dir = _collect_evidence(output_dir, build_dir, archive,
                                         headers, debug_dir)
    payload = _payload_manifest(args, repo_root, output_dir, row, revision,
                                snapshot, extension, plugin, metadata_path,
                                groups, evidence_dir)
    write_json(output_dir / "payload.json", payload)
    shutil.copy2(metadata_path, output_dir / "cutedsl-metadata.json")
    if source_snapshot(repo_root) != snapshot:
        raise RuntimeError(
            "Payload packaging modified tracked source or submodule state.")
    print(output_dir / "payload.json")


if __name__ == "__main__":
    main()
