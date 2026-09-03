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
"""Verify and safely extract the exact CuTe artifact for one qualified variant."""

from __future__ import annotations

import argparse
import re
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .config import REPO_ROOT, load_matrix, require_variant, sha256


def _arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--destination",
                        type=Path,
                        default=REPO_ROOT / "cpp" / "kernels" /
                        "cuteDSLArtifact")
    parser.add_argument("--matrix",
                        type=Path,
                        default=REPO_ROOT / "packaging" / "variants.toml")
    return parser.parse_args(argv)


def _artifact_name(row: Mapping[str, Any]) -> str:
    cuda_major = str(row["cuda_ctk_version"]).split(".", 1)[0]
    return (f"cutedsl_{row['cpu_arch']}_{row['cute_dsl_artifact_tag']}_"
            f"cuda{cuda_major}.tar.gz")


def _verified_archive(artifact_dir: Path, name: str) -> Path:
    archive_path = artifact_dir.resolve() / name
    checksum_path = archive_path.with_name(name + ".sha256")
    missing = [
        path for path in (archive_path, checksum_path) if not path.is_file()
    ]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        raise RuntimeError(
            "CuTe DSL artifact input is incomplete. Missing: "
            f"{missing_paths}. Generate the matching archive and checksum "
            "with kernelSrcs/build_cutedsl_tarballs.sh.")
    checksum = checksum_path.read_text(encoding="utf-8").strip().split()
    if (len(checksum) != 2 or checksum[1].lstrip("*") != name
            or not re.fullmatch(r"[0-9a-f]{64}", checksum[0])):
        raise RuntimeError(f"Invalid checksum file {checksum_path}.")
    if sha256(archive_path) != checksum[0]:
        raise RuntimeError(f"Checksum mismatch for {archive_path}.")
    return archive_path


def _validate_archive_members(members: Sequence[tarfile.TarInfo],
                              root: str) -> None:
    for member in members:
        path = PurePosixPath(member.name)
        if (path.is_absolute() or ".." in path.parts or not path.parts
                or path.parts[0] != root or member.issym() or member.islnk()
                or not (member.isfile() or member.isdir())):
            raise RuntimeError(f"Unsafe CuTe archive member {member.name!r}.")


def _extract_members(archive: tarfile.TarFile,
                     members: Sequence[tarfile.TarInfo], target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for member in members:
        destination = target.joinpath(*PurePosixPath(member.name).parts)
        if member.isdir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise RuntimeError(
                f"Cannot read CuTe archive member {member.name!r}.")
        with source, destination.open("wb") as output:
            shutil.copyfileobj(source, output)
        destination.chmod(member.mode & 0o777)


def _validate_extracted_artifact(required: Path, row: Mapping[str,
                                                              Any]) -> None:
    expected = (
        "metadata.json",
        f"libcutedsl_{row['cpu_arch']}.a",
        "include/cutedsl_all.h",
    )
    for relative in expected:
        if not (required / relative).is_file():
            raise RuntimeError(
                f"Extracted CuTe artifact is missing {relative}.")


def prepare_artifact(variant: str, artifact_dir: Path, destination: Path,
                     matrix_path: Path) -> Path:
    """Verify and extract the CuTe archive for one qualified variant."""
    _, rows = load_matrix(matrix_path.resolve())
    row = require_variant(rows, variant)
    archive_path = _verified_archive(artifact_dir, _artifact_name(row))
    root = str(row["cute_dsl_artifact_tag"])
    target = destination.resolve() / str(row["cpu_arch"])
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        _validate_archive_members(members, root)
        _extract_members(archive, members, target)
    required = target / root
    _validate_extracted_artifact(required, row)
    return required


def main(argv=None) -> None:
    args = _arguments(argv)
    required = prepare_artifact(args.variant, args.artifact_dir,
                                args.destination, args.matrix)
    print(required)


if __name__ == "__main__":
    main()
