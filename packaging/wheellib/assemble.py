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
"""Assemble verified payloads into one deterministic architecture/ABI wheel."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from .config import (CONTRACT, REPO_ROOT, load_matrix, load_toml, run_checked,
                     sha256, write_json)
from .verify import verify

_RUNTIME_FIELDS = set(CONTRACT.RUNTIME_VARIANT_FIELDS)


def _arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-wheel", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--cpu-arch",
                        choices=("x86_64", "aarch64"),
                        required=True)
    parser.add_argument("--python-abi", required=True)
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        help=
        "Include one variant; repeat for a subset wheel. Omit for the complete architecture."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrix",
                        type=Path,
                        default=REPO_ROOT / "packaging" / "variants.toml")
    parser.add_argument("--allowlist",
                        type=Path,
                        default=REPO_ROOT / "packaging" /
                        "dependency_allowlist.toml")
    parser.add_argument("--size-budget",
                        type=Path,
                        default=REPO_ROOT / "packaging" / "size_budget.toml")
    parser.add_argument(
        "--no-device-image-check",
        action="store_true",
        help="Development-only; do not use for release artifacts.")
    parser.add_argument(
        "--expected-sha256",
        help="Fail unless the assembled wheel has this digest.")
    return parser.parse_args(argv)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Cannot read assembly metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Assembly metadata must be an object: {path}.")
    return value


def _base_metadata(base_wheel: Path) -> Dict[str, Any]:
    metadata_path = base_wheel.parent / "base-wheel.json"
    metadata = _read_json(metadata_path)
    required = {
        "schema_version", "package_version", "source_revision", "wheel",
        "wheel_sha256"
    }
    required.update({"source_provenance_sha256", "submodule_revisions"})
    if metadata.get("schema_version") != 1 or not required.issubset(metadata):
        raise RuntimeError(f"Invalid base wheel metadata: {metadata_path}.")
    if metadata["wheel"] != base_wheel.name or metadata[
            "wheel_sha256"] != sha256(base_wheel):
        raise RuntimeError(
            "Base wheel name or SHA-256 does not match base-wheel.json.")
    with zipfile.ZipFile(base_wheel) as archive:
        forbidden = [
            name for name in archive.namelist()
            if name.endswith((".so", ".a", ".o", ".cubin", ".fatbin"))
            or name.endswith("/_native/variants.json")
        ]
    if forbidden:
        raise RuntimeError(
            f"Base wheel already contains native payload data: {forbidden}.")
    return metadata


def _copy_payload(stage: Path, unpacked_root: Path, variant_id: str) -> None:
    source = stage / "tensorrt_edgellm" / "_native" / "payloads" / variant_id
    destination = unpacked_root / "tensorrt_edgellm" / "_native" / "payloads" / variant_id
    if destination.exists():
        raise RuntimeError(f"Duplicate payload destination {destination}.")
    shutil.copytree(source,
                    destination,
                    symlinks=False,
                    ignore=shutil.ignore_patterns("licenses"))


def _set_wheel_tags(unpacked_root: Path, python_abi: str,
                    cpu_arch: str) -> None:
    dist_infos = list(unpacked_root.glob("*.dist-info"))
    if len(dist_infos) != 1:
        raise RuntimeError(
            f"Expected one .dist-info directory, found {dist_infos}.")
    wheel_metadata = dist_infos[0] / "WHEEL"
    lines = [
        line
        for line in wheel_metadata.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(("Tag:", "Root-Is-Purelib:"))
    ]
    lines.extend([
        "Root-Is-Purelib: false",
        f"Tag: {python_abi}-{python_abi}-linux_{cpu_arch}",
    ])
    wheel_metadata.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normalize_mtimes(root: Path) -> None:
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "315532800"))
    for path in sorted(root.rglob("*")):
        os.utime(path, (epoch, epoch), follow_symlinks=False)
    os.utime(root, (epoch, epoch), follow_symlinks=False)


def _wheel_members(archive: zipfile.ZipFile) -> Tuple[Set[str], str, str]:
    raw_names = archive.namelist()
    if len(raw_names) != len(set(raw_names)):
        raise RuntimeError("Final wheel contains duplicate ZIP members.")
    names = set(raw_names)
    wheel_files = [name for name in names if name.endswith(".dist-info/WHEEL")]
    record_files = [
        name for name in names if name.endswith(".dist-info/RECORD")
    ]
    if len(wheel_files) != 1 or len(record_files) != 1:
        raise RuntimeError(
            "Final wheel must contain exactly one WHEEL and RECORD file.")
    return names, wheel_files[0], record_files[0]


def _audit_wheel_metadata(archive: zipfile.ZipFile, wheel_file: str,
                          cpu_arch: str, python_abi: str) -> None:
    wheel_metadata = archive.read(wheel_file).decode()
    tag = f"Tag: {python_abi}-{python_abi}-linux_{cpu_arch}"
    if tag not in wheel_metadata or "Root-Is-Purelib: false" not in wheel_metadata:
        raise RuntimeError(
            f"Final WHEEL metadata is missing {tag!r} or native-root marker.")


def _load_runtime_manifest(archive: zipfile.ZipFile,
                           names: Set[str]) -> Dict[str, Any]:
    manifest_name = "tensorrt_edgellm/_native/variants.json"
    if manifest_name not in names:
        raise RuntimeError("Final wheel has no generated variants.json.")
    manifest = json.loads(archive.read(manifest_name))
    try:
        return CONTRACT.validate_manifest(manifest)
    except ValueError as error:
        raise RuntimeError(
            f"Generated variants.json fails its contract: {error}") from error


def _audit_manifest_entries(manifest: Dict[str, Any], names: Set[str],
                            expected_ids: Iterable[str]) -> None:
    expected = set(expected_ids)
    actual = {entry["variant_id"] for entry in manifest.get("variants", [])}
    if actual != expected:
        raise RuntimeError(
            f"Final manifest payload set differs: {actual} != {expected}.")
    for entry in manifest["variants"]:
        if set(entry) != _RUNTIME_FIELDS:
            raise RuntimeError(
                f"Runtime entry {entry.get('variant_id')} has schema drift: "
                f"{sorted(set(entry) ^ _RUNTIME_FIELDS)}.")
        for key in ("extension", "plugin"):
            installed = f"tensorrt_edgellm/{entry[key]}"
            if installed not in names:
                raise RuntimeError(
                    f"Manifest path is absent from wheel: {installed}.")


def _audit_user_packages(archive: zipfile.ZipFile, names: Set[str]) -> None:
    required = {
        "tensorrt_edgellm/__init__.py",
        "tensorrt_edgellm/runtime.py",
        "experimental/builder/__init__.py",
        "experimental/builder/cli.py",
    }
    missing = sorted(required - names)
    entry_points = [
        name for name in names if name.endswith(".dist-info/entry_points.txt")
    ]
    if missing or len(entry_points) != 1:
        raise RuntimeError(
            f"Final wheel is missing build/runtime package files: {missing}.")
    content = archive.read(entry_points[0]).decode("utf-8")
    if "tensorrt-edgellm-build = experimental.builder.cli:main" not in content:
        raise RuntimeError(
            "Final wheel has no tensorrt-edgellm-build entry point.")


def _audit_wheel_contents(names: Set[str]) -> None:
    forbidden = [
        name for name in names
        if name.endswith((".a", ".o", ".map", ".bin")) or "/evidence/" in name
    ]
    if forbidden:
        raise RuntimeError(
            f"Final wheel contains build-only native files: {forbidden}.")


def _audit_record(archive: zipfile.ZipFile, names: Set[str],
                  record_file: str) -> None:
    record_rows = list(
        csv.reader(io.StringIO(archive.read(record_file).decode("utf-8"))))
    if any(len(row) != 3 for row in record_rows):
        raise RuntimeError("Wheel RECORD contains a malformed row.")
    records = {row[0]: (row[1], row[2]) for row in record_rows}
    if len(records) != len(record_rows) or names != set(records):
        raise RuntimeError(
            "Wheel RECORD paths do not exactly cover archive members.")
    for name, (digest, size) in records.items():
        if name == record_file:
            if digest or size:
                raise RuntimeError(
                    "The RECORD self-entry must have empty hash and size.")
            continue
        data = archive.read(name)
        encoded = base64.urlsafe_b64encode(
            hashlib.sha256(data).digest()).rstrip(b"=").decode()
        if digest != f"sha256={encoded}" or size != str(len(data)):
            raise RuntimeError(
                f"Wheel RECORD digest or size mismatch for {name}.")


def _audit_licenses(archive: zipfile.ZipFile, names: Set[str]) -> None:
    licenses = [name for name in names if ".dist-info/licenses/" in name]
    required = {
        "LICENSE": REPO_ROOT / "LICENSE",
        "LICENSE.MIT": REPO_ROOT / "3rdParty" / "nlohmannJson" / "LICENSE.MIT",
    }
    if {Path(name).name for name in licenses} != set(required):
        raise RuntimeError(
            "Final wheel license inventory must contain the project and nlohmann notices."
        )
    for name in licenses:
        expected = required[Path(name).name]
        if hashlib.sha256(archive.read(name)).hexdigest() != sha256(expected):
            raise RuntimeError(f"Final wheel license differs from {expected}.")


def _audit_unpacked_size(archive: zipfile.ZipFile,
                         unpacked_budget: int) -> None:
    unpacked = sum(info.file_size for info in archive.infolist())
    if unpacked > unpacked_budget:
        raise RuntimeError(
            f"Wheel unpacked size {unpacked} exceeds reviewed budget {unpacked_budget}."
        )


def _audit_final_wheel(wheel: Path, cpu_arch: str, python_abi: str,
                       expected_ids: Iterable[str],
                       unpacked_budget: int) -> None:
    """Validate final tags, manifest, RECORD coverage, content, and size."""
    expected_suffix = f"-{python_abi}-{python_abi}-linux_{cpu_arch}.whl"
    if not wheel.name.endswith(expected_suffix):
        raise RuntimeError(
            f"Final wheel filename does not end with {expected_suffix}.")
    with zipfile.ZipFile(wheel) as archive:
        names, wheel_file, record_file = _wheel_members(archive)
        _audit_wheel_metadata(archive, wheel_file, cpu_arch, python_abi)
        manifest = _load_runtime_manifest(archive, names)
        _audit_manifest_entries(manifest, names, expected_ids)
        _audit_user_packages(archive, names)
        _audit_wheel_contents(names)
        _audit_record(archive, names, record_file)
        _audit_licenses(archive, names)
        _audit_unpacked_size(archive, unpacked_budget)


def _selected_variant_ids(rows: Iterable[Dict[str, Any]], cpu_arch: str,
                          requested: List[str]) -> Tuple[Set[str], Set[str]]:
    architecture_ids = {
        str(row["variant_id"])
        for row in rows if row["cpu_arch"] == cpu_arch
    }
    if not architecture_ids:
        raise RuntimeError(f"The matrix has no variants for {cpu_arch}.")
    if not requested:
        return architecture_ids, architecture_ids
    selected = set(requested)
    if len(selected) != len(requested):
        raise RuntimeError("Each requested variant must be unique.")
    invalid = selected - architecture_ids
    if invalid:
        raise RuntimeError(
            f"Variants are not qualified for {cpu_arch}: {sorted(invalid)}.")
    return selected, architecture_ids


def _tag_subset_wheel(wheel: Path, python_abi: str, selected: Set[str],
                      architecture: Set[str]) -> Path:
    if selected == architecture:
        return wheel
    digest = hashlib.sha256("\n".join(sorted(selected)).encode()).hexdigest()
    marker = f"-{python_abi}-{python_abi}-"
    if wheel.name.count(marker) != 1:
        raise RuntimeError(f"Cannot add a subset build tag to {wheel.name}.")
    tagged_name = wheel.name.replace(marker, f"-1subset{digest[:8]}{marker}",
                                     1)
    tagged = wheel.with_name(tagged_name)
    wheel.rename(tagged)
    return tagged


def main(argv=None) -> None:
    """Verify selected payloads, inject them, retag them, and pack one wheel."""
    args = _arguments(argv)
    base_wheel = args.base_wheel.resolve(strict=True)
    base = _base_metadata(base_wheel)
    matrix, rows = load_matrix(args.matrix.resolve())
    if args.python_abi not in matrix["qualified_python_abis"]:
        raise RuntimeError(f"Unqualified Python ABI {args.python_abi!r}.")
    expected_ids, architecture_ids = _selected_variant_ids(
        rows, args.cpu_arch, args.variants or [])
    payloads: List[Dict[str, Any]] = []
    stages: Dict[str, Path] = {}
    for metadata_path in sorted(
            args.payload_root.resolve().rglob("payload.json")):
        stage = metadata_path.parent
        payload = _read_json(metadata_path)
        if (payload.get("cpu_arch") != args.cpu_arch
                or payload.get("python_abi") != args.python_abi
                or str(payload.get("variant_id")) not in expected_ids):
            continue
        verified = verify(stage,
                          args.allowlist,
                          args.size_budget,
                          matrix_path=args.matrix,
                          require_device_images=not args.no_device_image_check,
                          require_dependency_resolution=False)
        variant_id = str(verified["variant_id"])
        if variant_id in stages:
            raise RuntimeError(f"Duplicate payload stage for {variant_id}.")
        stages[variant_id] = stage
        payloads.append(verified)
    found_ids = set(stages)
    if found_ids != expected_ids:
        raise RuntimeError(
            f"Incomplete {args.cpu_arch}/{args.python_abi} fan-in: "
            f"missing={sorted(expected_ids - found_ids)}, extra={sorted(found_ids - expected_ids)}."
        )
    for payload in payloads:
        if payload["source_revision"] != base["source_revision"]:
            raise RuntimeError(
                f"Revision mismatch in payload {payload['variant_id']}.")
        if payload["package_version"] != base["package_version"]:
            raise RuntimeError(
                f"Version mismatch in payload {payload['variant_id']}.")
        if payload["source_provenance_sha256"] != base[
                "source_provenance_sha256"]:
            raise RuntimeError(
                f"Source provenance mismatch in payload {payload['variant_id']}."
            )
        if payload["submodule_revisions"] != base["submodule_revisions"]:
            raise RuntimeError(
                f"Submodule revision mismatch in payload {payload['variant_id']}."
            )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if list(output_dir.glob("*.whl")):
        raise RuntimeError(
            f"Assembly output directory must contain no wheels: {output_dir}.")
    with tempfile.TemporaryDirectory(prefix="edgellm-assemble-") as temp:
        temp_root = Path(temp)
        run_checked([
            "wheel", "unpack", "--dest",
            os.fspath(temp_root),
            os.fspath(base_wheel)
        ])
        unpacked = [path for path in temp_root.iterdir() if path.is_dir()]
        if len(unpacked) != 1:
            raise RuntimeError(
                f"wheel unpack produced unexpected directories: {unpacked}.")
        root = unpacked[0]
        for payload in sorted(payloads, key=lambda value: value["variant_id"]):
            _copy_payload(stages[payload["variant_id"]], root,
                          payload["variant_id"])
        runtime_entries = [{
            key: payload.get(key)
            for key in sorted(_RUNTIME_FIELDS)
        } for payload in sorted(payloads,
                                key=lambda value: value["variant_id"])]
        for entry in runtime_entries:
            prefix = f"_native/payloads/{entry['variant_id']}/"
            if entry["plugin"] != prefix + "lib/libNvInfer_edgellm_plugin.so":
                raise RuntimeError(
                    f"Variant {entry['variant_id']} has an invalid plugin path."
                )
        write_json(
            root / "tensorrt_edgellm" / "_native" / "variants.json", {
                "schema_version": 1,
                "package_version": base["package_version"],
                "source_revision": base["source_revision"],
                "source_provenance_sha256": base["source_provenance_sha256"],
                "variants": runtime_entries,
            })
        _set_wheel_tags(root, args.python_abi, args.cpu_arch)
        _normalize_mtimes(root)
        before = set(output_dir.glob("*.whl"))
        environment = dict(os.environ)
        environment.setdefault("SOURCE_DATE_EPOCH", "315532800")
        run_checked([
            "wheel", "pack", "--dest-dir",
            os.fspath(output_dir),
            os.fspath(root)
        ],
                    env=environment)
        produced = list(set(output_dir.glob("*.whl")) - before)
        if len(produced) != 1:
            raise RuntimeError(
                f"Expected one assembled wheel, found {produced}.")
    wheel = _tag_subset_wheel(produced[0], args.python_abi, expected_ids,
                              architecture_ids)
    budgets = load_toml(args.size_budget)["wheel"][args.cpu_arch]
    try:
        if wheel.stat().st_size > int(budgets["compressed_bytes"]):
            raise RuntimeError(
                f"Wheel {wheel} exceeds compressed size budget.")
        _audit_final_wheel(wheel, args.cpu_arch, args.python_abi, expected_ids,
                           int(budgets["unpacked_bytes"]))
        digest = sha256(wheel)
        if args.expected_sha256 and digest != args.expected_sha256:
            raise RuntimeError(
                f"Reproducibility digest mismatch: {digest} != {args.expected_sha256}."
            )
    except Exception:
        wheel.unlink(missing_ok=True)
        raise
    print(f"sha256:{digest}")
    print(wheel)


if __name__ == "__main__":
    main()
