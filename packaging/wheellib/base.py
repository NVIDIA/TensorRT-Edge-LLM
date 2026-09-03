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
"""Build the canonical native-free Python wheel used for native fan-in."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from .config import (REPO_ROOT, package_version, require_clean_source,
                     run_checked, sha256, source_revision, source_snapshot,
                     write_json)


def _arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",
                        type=Path,
                        required=True,
                        help="Directory for the wheel and base-wheel.json.")
    parser.add_argument("--repo-root",
                        type=Path,
                        default=REPO_ROOT,
                        help="EdgeLLM source root (default: repository root).")
    parser.add_argument("--source-revision",
                        help="Expected full Git revision; defaults to HEAD.")
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help="Development-only override; do not use for release artifacts.")
    return parser.parse_args(argv)


def _verify_native_free(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        forbidden = []
        for info in archive.infolist():
            name = info.filename
            if (name.endswith(
                (".so", ".a", ".o", ".cubin", ".fatbin", ".cpp", ".cu",
                 ".cuh")) or name.endswith("/_native/variants.json")
                    or name.startswith("tensorrt_edgellm/_native/payloads/")
                    or "/experimental/pybind/" in f"/{name}"
                    or archive.read(info).startswith(b"\x7fELF")):
                forbidden.append(name)
    if forbidden:
        raise RuntimeError(
            "Base wheel contains native/generated/source files reserved for "
            f"payload assembly: {', '.join(forbidden)}.")


def main(argv=None) -> None:
    """Build one Python-only wheel and write its revision-bound metadata."""
    args = _arguments(argv)
    repo_root = args.repo_root.resolve()
    revision = args.source_revision or source_revision(repo_root)
    if revision != source_revision(repo_root):
        raise RuntimeError(
            f"Requested revision {revision} does not match checkout HEAD.")
    if not args.allow_dirty_source:
        require_clean_source(repo_root)
    snapshot = source_snapshot(repo_root)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="edgellm-base-wheel-") as temp:
        temporary_output = Path(temp)
        environment = dict(os.environ)
        environment["SKBUILD_CMAKE_ARGS"] = "-DEDGELLM_PYTHON_ONLY_WHEEL=ON"
        run_checked([
            os.fspath(Path(os.sys.executable)),
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            os.fspath(temporary_output),
            os.fspath(repo_root),
        ],
                    env=environment)
        wheels = list(temporary_output.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(
                f"Expected one base wheel, found {len(wheels)} in {temporary_output}."
            )
        _verify_native_free(wheels[0])
        wheel = output_dir / wheels[0].name
        shutil.copy2(wheels[0], wheel)
    write_json(
        output_dir / "base-wheel.json", {
            "schema_version": 1,
            "package_version": package_version(repo_root),
            "source_revision": revision,
            "wheel": wheel.name,
            "wheel_sha256": sha256(wheel),
            "source_provenance_sha256": snapshot["sha256"],
            "submodule_revisions": snapshot["submodule_revisions"],
        })
    if source_snapshot(repo_root) != snapshot:
        raise RuntimeError(
            "Base-wheel packaging modified tracked source or submodule state.")
    print(wheel)


if __name__ == "__main__":
    main()
