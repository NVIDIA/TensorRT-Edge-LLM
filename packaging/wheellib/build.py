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
"""Build a selected-target wheel or assemble a complete architecture wheel."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from . import assemble, base, cutedsl, payload, verify
from .config import REPO_ROOT, load_matrix


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--local",
        action="store_true",
        help=
        "Build the unique matrix variant matching the visible GPU and host.")
    selection.add_argument(
        "--variant",
        action="append",
        dest="variants",
        help="Build one variant; repeat for compatible variants in one wheel.")
    selection.add_argument(
        "--all-for-arch",
        choices=("x86_64", "aarch64"),
        help="Assemble every architecture variant from --payload-root.")
    parser.add_argument(
        "--trt-package-dir",
        type=Path,
        help="TensorRT SDK used for local or selected-variant payload builds.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=REPO_ROOT / "kernelSrcs" / "cuteDSLPrebuilt",
        help="Directory containing CuTe DSL archives and checksum files.")
    parser.add_argument(
        "--payload-root",
        type=Path,
        help="Existing payload root required by --all-for-arch.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "source-wheel",
        help=
        "Fresh directory for generated base and selected payload artifacts.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--build-jobs", type=_positive_int)
    parser.add_argument("--toolchain-file", type=Path)
    parser.add_argument("--target-sysroot", type=Path)
    parser.add_argument("--target-python-include-dir", type=Path)
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="Development-only override for a checkout with tracked changes.")
    parser.add_argument(
        "--no-device-image-check",
        action="store_true",
        help="Development-only override for payload device-image verification."
    )
    parser.add_argument("--matrix",
                        type=Path,
                        default=REPO_ROOT / "packaging" / "variants.toml")
    return parser.parse_args(argv)


def _python_abi() -> str:
    tag = getattr(sys.implementation, "cache_tag", "") or ""
    match = re.fullmatch(r"cpython-(\d{2,3})", tag)
    if match is None:
        raise RuntimeError(f"Unsupported build interpreter cache tag {tag!r}.")
    return f"cp{match.group(1)}"


def _host_arch() -> str:
    machine = platform.machine().lower()
    return {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)


def _probe_matches(row: Mapping[str, Any], detected: Any) -> bool:
    if row["platform_probe_source"] != detected.platform_probe_source:
        return False
    values = row["platform_probe_values"]
    if detected.platform_probe_source == "device-model":
        model = detected.platform_probe_value.casefold()
        return any(str(value).casefold() in model for value in values)
    return detected.platform_probe_value in values


def _detect_platform() -> Any:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    module = importlib.import_module("tensorrt_edgellm._native.detect")
    return module.detect_platform()


def _local_variant(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    detected = _detect_platform()
    matches = [
        row for row in rows if row["cpu_arch"] == detected.cpu_arch
        and row["cuda_runtime_soname"] == detected.cuda_runtime_soname
        and row["tensorrt_runtime_soname"] == detected.tensorrt_runtime_soname
        and int(row["gpu_sm"]) == detected.gpu_sm
        and _probe_matches(row, detected)
    ]
    if len(matches) != 1:
        names = sorted(str(row["variant_id"]) for row in matches)
        raise RuntimeError(
            "Expected one public build variant for the detected platform, "
            f"found {names or 'none'} for {detected.as_dict()}.")
    return dict(matches[0])


def _requested_variants(rows: Sequence[Mapping[str, Any]],
                        names: Sequence[str]) -> List[Dict[str, Any]]:
    if len(set(names)) != len(names):
        raise RuntimeError("Each requested variant must be unique.")
    by_id = {str(row["variant_id"]): dict(row) for row in rows}
    missing = sorted(set(names) - set(by_id))
    if missing:
        raise RuntimeError(f"Unknown public wheel variants: {missing}.")
    return [by_id[name] for name in names]


def _build_context(row: Mapping[str, Any]) -> Tuple[object, ...]:
    return (
        row["cpu_arch"],
        row["platform_family"],
        row["platform_release"],
        row["cuda_runtime_soname"],
        row["tensorrt_runtime_soname"],
        row["cuda_ctk_version"],
        row["embedded_target"],
        tuple(row["cmake_args"]),
    )


def _validate_selected_context(rows: Sequence[Mapping[str, Any]]) -> None:
    contexts = {_build_context(row) for row in rows}
    if len(contexts) != 1:
        raise RuntimeError(
            "Selected variants require different platform, CUDA, TensorRT, or "
            "toolchain environments. Build their payloads separately and use "
            "--all-for-arch with the resulting payload root.")


def _clean_work_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise RuntimeError(f"Wheel work directory must be empty: {resolved}.")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _base_arguments(args: argparse.Namespace, output: Path) -> List[str]:
    values = ["--output-dir", str(output)]
    if args.allow_dirty_source:
        values.append("--allow-dirty-source")
    return values


def _payload_arguments(args: argparse.Namespace, variant: str, python_abi: str,
                       output: Path) -> List[str]:
    if args.trt_package_dir is None:
        raise RuntimeError("--trt-package-dir is required.")
    values = [
        "--variant",
        variant,
        "--python-abi",
        python_abi,
        "--trt-package-dir",
        str(args.trt_package_dir),
        "--output-dir",
        str(output),
        "--matrix",
        str(args.matrix),
    ]
    optional_paths = (
        ("--toolchain-file", args.toolchain_file),
        ("--target-sysroot", args.target_sysroot),
        ("--target-python-include-dir", args.target_python_include_dir),
    )
    for option, value in optional_paths:
        if value is not None:
            values.extend([option, str(value)])
    if args.build_jobs is not None:
        values.extend(["--build-jobs", str(args.build_jobs)])
    if args.allow_dirty_source:
        values.append("--allow-dirty-source")
    return values


def _verify_payload(args: argparse.Namespace, stage: Path) -> None:
    values = [
        "--stage",
        str(stage),
        "--matrix",
        str(args.matrix),
        "--dependency-root",
        str(args.trt_package_dir),
    ]
    if args.target_sysroot is not None:
        values.extend(["--dependency-root", str(args.target_sysroot)])
    if args.no_device_image_check:
        values.append("--no-device-image-check")
    verify.main(values)


def _build_selected_payloads(args: argparse.Namespace,
                             rows: Sequence[Mapping[str, Any]],
                             python_abi: str, work_dir: Path) -> Path:
    if args.trt_package_dir is None:
        raise RuntimeError(
            "--trt-package-dir is required with --local or --variant.")
    _validate_selected_context(rows)
    payload_root = work_dir / "payloads"
    for row in rows:
        variant = str(row["variant_id"])
        cutedsl.main([
            "--variant",
            variant,
            "--artifact-dir",
            str(args.artifact_dir),
            "--matrix",
            str(args.matrix),
        ])
        stage = payload_root / f"{variant}-{python_abi}"
        payload.main(_payload_arguments(args, variant, python_abi, stage))
        _verify_payload(args, stage)
    return payload_root


def _base_wheel(args: argparse.Namespace, python_abi: str,
                work_dir: Path) -> Path:
    output = work_dir / "base" / python_abi
    base.main(_base_arguments(args, output))
    metadata = json.loads(
        (output / "base-wheel.json").read_text(encoding="utf-8"))
    wheel = output / str(metadata["wheel"])
    return wheel.resolve(strict=True)


def _assemble_arguments(args: argparse.Namespace, base_wheel: Path,
                        payload_root: Path, cpu_arch: str, python_abi: str,
                        variants: Sequence[str]) -> List[str]:
    values = [
        "--base-wheel",
        str(base_wheel),
        "--payload-root",
        str(payload_root),
        "--cpu-arch",
        cpu_arch,
        "--python-abi",
        python_abi,
        "--output-dir",
        str(args.output_dir),
        "--matrix",
        str(args.matrix),
    ]
    for variant in variants:
        values.extend(["--variant", variant])
    if args.no_device_image_check:
        values.append("--no-device-image-check")
    return values


def main(argv: Sequence[str] | None = None) -> None:
    """Build selected payloads or assemble a complete architecture wheel."""
    args = _arguments(argv)
    matrix, all_rows = load_matrix(args.matrix.resolve())
    python_abi = _python_abi()
    if python_abi not in matrix["qualified_python_abis"]:
        raise RuntimeError(f"Unqualified build interpreter ABI {python_abi}.")
    work_dir = _clean_work_dir(args.work_dir)
    if args.all_for_arch:
        if args.payload_root is None:
            raise RuntimeError(
                "--payload-root is required with --all-for-arch because the "
                "architecture payloads use target-specific build environments."
            )
        cpu_arch = args.all_for_arch
        selected_names: List[str] = []
        payload_root = args.payload_root.resolve(strict=True)
    else:
        selected = ([_local_variant(all_rows)] if args.local else
                    _requested_variants(all_rows, args.variants or []))
        cpu_arch = str(selected[0]["cpu_arch"])
        if cpu_arch != _host_arch() and args.toolchain_file is None:
            raise RuntimeError(
                f"Building {cpu_arch} from {_host_arch()} requires --toolchain-file."
            )
        selected_names = [str(row["variant_id"]) for row in selected]
        payload_root = _build_selected_payloads(args, selected, python_abi,
                                                work_dir)
    base_wheel = _base_wheel(args, python_abi, work_dir)
    assemble.main(
        _assemble_arguments(args, base_wheel, payload_root, cpu_arch,
                            python_abi, selected_names))
