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
"""Produce wheel-only CuTe artifacts from sanitized kernel sources."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from . import config, cutedsl, oss


def main(argv=None) -> None:
    """Build the wheel kernel matrix without reusing internal kernel artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument(
        "--variant",
        action="append",
        help="Generate only these qualified variants; default: full matrix.")
    args = parser.parse_args(argv)
    root = config.REPO_ROOT
    if not args.allow_dirty_source:
        config.require_clean_source(root)
    _, rows = config.load_matrix(root / "packaging/variants.toml")
    if args.variant:
        rows = [config.require_variant(rows, name) for name in args.variant]
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"OSS kernel output directory must be empty: {output}.")
    output.mkdir(parents=True, exist_ok=True)
    targets = sorted({(str(row["cpu_arch"]), str(row["cute_dsl_artifact_tag"]),
                       str(row["cuda_ctk_version"]).split(".")[0])
                      for row in rows})
    with tempfile.TemporaryDirectory(prefix="edgellm-oss-kernels-") as temp:
        stage = oss.stage_source(root,
                                 Path(temp) / "source",
                                 include_submodules=False)
        kernel_digest = oss.tree_digest(stage / "kernelSrcs")
        environment = dict(os.environ)
        environment.update({
            "CUTE_DSL_MATRIX":
            ",".join(f"{arch}:{tag.replace('_sm_', '+sm_')}:{cuda}"
                     for arch, tag, cuda in targets),
            "CUTE_DSL_OUTPUT_DIR":
            str(Path(temp) / "artifacts"),
            "CUTE_DSL_PREBUILT_DIR":
            str(output),
            "CUTE_DSL_KERNELS":
            "ALL",
        })
        config.run_checked(
            ["bash",
             str(stage / "kernelSrcs/build_cutedsl_tarballs.sh")],
            cwd=stage,
            env=environment)
        for arch, tag, cuda in targets:
            artifact = Path(temp) / "artifacts" / f"cuda{cuda}" / arch / tag
            for path in artifact.rglob("*"):
                if path.is_file():
                    oss.audit_file(path, root)
            archive = cutedsl._verified_archive(
                output, f"cutedsl_{arch}_{tag}_cuda{cuda}.tar.gz")
            config.write_json(
                archive.with_name(archive.name + ".oss.json"), {
                    "oss_policy_sha256": oss.policy_digest(root),
                    "kernel_source_sha256": kernel_digest,
                    "archive_sha256": config.sha256(archive),
                    "artifact_tree_sha256": oss.tree_digest(artifact),
                })
