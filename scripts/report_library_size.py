#!/usr/bin/env python3
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
"""Report the size of the built Edge-LLM libraries.

Prints a table of the produced artifacts and, for the plugin shared library, a
breakdown of where the bytes went (XQA embedded cubins, CuTe DSL AOT kernel
blobs, nvcc fatbin, host code). Optionally writes a JSON summary and an
OpenMetrics report so every CI build records its library size.

The ELF is parsed directly, so this works on a cross-build machine without
needing target binutils.

The oldest CI image (DriveOS 6 / CUDA 11.4) still ships Python 3.8, so keep
this script compatible with it; the __future__ import below is what lets the
`dict[str, int]` / `Path | None` annotations be used there.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

# Section names holding nvcc-generated device code (no symbols point at them).
_FATBIN_SECTIONS = (".nv_fatbin", "__nv_relfatbin", ".nvFatBinSegment")

# Artifacts reported for every build, relative to the build directory.
_ARTIFACTS = (
    "libNvInfer_edgellm_plugin.so.1.0",
    "cpp/libedgellmCore.a",
    "cpp/libedgellmKernels.a",
    "cpp/libedgellmBuilder.a",
    "examples/llm/llm_build",
    "examples/llm/llm_inference",
    "examples/llm/llm_stream",
    "examples/llm/llm_bench",
    "unittests/unitTestRuntime",
)


class ElfError(RuntimeError):
    """Raised when a file is not a 64-bit little-endian ELF we can parse."""


class Elf:
    """Minimal 64-bit little-endian ELF reader (sections + symbol table)."""

    def __init__(self, path: Path):
        self.path = path
        self.blob = path.read_bytes()
        if self.blob[:4] != b"\x7fELF":
            raise ElfError(f"{path} is not an ELF file")
        if self.blob[4] != 2 or self.blob[5] != 1:
            raise ElfError(f"{path} is not 64-bit little-endian ELF")
        shoff, = struct.unpack_from("<Q", self.blob, 0x28)
        shentsize, shnum, shstrndx = struct.unpack_from(
            "<HHH", self.blob, 0x3A)
        self._sections = [
            self._section(shoff + i * shentsize) for i in range(shnum)
        ]
        shstrtab = self._sections[shstrndx]
        for section in self._sections:
            section["name"] = self._string(shstrtab["offset"],
                                           section["name_off"])

    def _section(self, offset: int) -> dict:
        (name_off, sh_type, _flags, _addr, sh_offset, sh_size, sh_link, _info,
         _align, _entsize) = struct.unpack_from("<IIQQQQIIQQ", self.blob,
                                                offset)
        return {
            "name_off": name_off,
            "type": sh_type,
            "offset": sh_offset,
            "size": sh_size,
            "link": sh_link,
        }

    def _string(self, table_offset: int, index: int) -> str:
        start = table_offset + index
        end = self.blob.index(b"\0", start)
        return self.blob[start:end].decode("utf-8", "replace")

    def section_sizes(self) -> dict[str, int]:
        # SHT_NOBITS (8, i.e. .bss) occupies no file bytes.
        return {
            section["name"]: section["size"]
            for section in self._sections if section["type"] != 8
        }

    def symbols(self):
        """Yield (name, size) for every sized symbol in .symtab."""
        for section in self._sections:
            if section["type"] != 2:  # SHT_SYMTAB
                continue
            strtab = self._sections[section["link"]]
            for offset in range(section["offset"],
                                section["offset"] + section["size"], 24):
                name_off, = struct.unpack_from("<I", self.blob, offset)
                size, = struct.unpack_from("<Q", self.blob, offset + 16)
                if size:
                    yield self._string(strtab["offset"], name_off), size

    def symbol_table_bytes(self) -> int:
        return sum(section["size"] for section in self._sections
                   if section["name"] in (".symtab", ".strtab"))


def classify(elf: Elf) -> dict[str, int]:
    """Split the shared library into its major byte consumers."""
    sections = elf.section_sizes()
    breakdown = {
        "xqa_cubins": 0,
        "cutedsl_kernels": 0,
        "nvcc_fatbin": sum(sections.get(name, 0) for name in _FATBIN_SECTIONS),
        "host_text": sections.get(".text", 0),
        "symbol_table": elf.symbol_table_bytes(),
        "other_symbols": 0,
    }
    for name, size in elf.symbols():
        if "kernels_binary" in name:
            breakdown["cutedsl_kernels"] += size
        elif "xqa" in name and "cubin" in name:
            breakdown["xqa_cubins"] += size
        else:
            breakdown["other_symbols"] += size
    return breakdown


def cutedsl_archive(build_dir: Path, source_dir: Path) -> Path | None:
    """Locate the CuTe DSL archive this build tree links, via CMakeCache."""
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return None
    tag = ""
    for line in cache.read_text(errors="replace").splitlines():
        if line.startswith("CUTE_DSL_ARTIFACT_TAG:"):
            tag = line.split("=", 1)[1].strip()
    if not tag:
        return None
    root = source_dir / "cpp" / "kernels" / "cuteDSLArtifact"
    matches = sorted(root.glob(f"*/{tag}/libcutedsl_*.a"))
    return matches[0] if matches else None


def collect(build_dir: Path, source_dir: Path) -> dict:
    report = {"artifacts": {}, "breakdown": {}, "cutedsl_archive": {}}

    for relative in _ARTIFACTS:
        path = build_dir / relative
        if path.is_file():
            report["artifacts"][relative] = path.stat().st_size

    plugin = build_dir / _ARTIFACTS[0]
    if plugin.is_file():
        try:
            report["breakdown"] = classify(Elf(plugin))
        except ElfError as error:
            print(f"warning: {error}", file=sys.stderr)

    archive = cutedsl_archive(build_dir, source_dir)
    if archive is not None:
        report["cutedsl_archive"] = {
            "path": str(archive.relative_to(source_dir)),
            "bytes": archive.stat().st_size,
        }
    return report


def render(report: dict, label: str) -> str:
    lines = [
        "==============================================================",
        f"Edge-LLM library size report — target: {label}",
        "==============================================================",
        "",
        "Artifacts:",
    ]
    for name, size in report["artifacts"].items():
        lines.append(f"  {size:>14,} B  {size / 2**20:>7.1f} MiB  {name}")

    breakdown = report["breakdown"]
    if breakdown:
        total = report["artifacts"].get(_ARTIFACTS[0], 0)
        lines += ["", f"Breakdown of {_ARTIFACTS[0]} ({total:,} B):"]
        for name, size in sorted(breakdown.items(), key=lambda kv: -kv[1]):
            share = f"{100 * size / total:5.1f} %" if total else "     - "
            lines.append(f"  {size:>14,} B  {share}  {name}")

    archive = report["cutedsl_archive"]
    if archive:
        lines += [
            "",
            f"Linked CuTe DSL archive: {archive['bytes']:,} B  "
            f"({archive['path']})",
        ]
    return "\n".join(lines) + "\n"


def metrics(report: dict, label: str) -> str:
    """OpenMetrics report of per-artifact and per-component sizes."""
    lines = [
        "# HELP edgellm_artifact_bytes Size of a built Edge-LLM artifact.",
        "# TYPE edgellm_artifact_bytes gauge",
    ]
    for name, size in report["artifacts"].items():
        lines.append(
            f'edgellm_artifact_bytes{{target="{label}",artifact="{name}"}} {size}'
        )
    lines += [
        "# HELP edgellm_component_bytes Bytes per component of the plugin library.",
        "# TYPE edgellm_component_bytes gauge",
    ]
    for name, size in report["breakdown"].items():
        lines.append(
            f'edgellm_component_bytes{{target="{label}",component="{name}"}} {size}'
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir",
                        type=Path,
                        default=Path("build"),
                        help="CMake build directory (default: build)")
    parser.add_argument("--source-dir",
                        type=Path,
                        default=Path("."),
                        help="Repository root (default: .)")
    parser.add_argument("--label",
                        default="native",
                        help="Target label used in the report and metrics")
    parser.add_argument("--output-dir",
                        type=Path,
                        help="Write lib_size.txt, lib_size.json and "
                        "lib_size_metrics.txt here")
    args = parser.parse_args()

    if not args.build_dir.is_dir():
        print(f"error: build directory not found: {args.build_dir}",
              file=sys.stderr)
        return 1

    report = collect(args.build_dir, args.source_dir)
    if not report["artifacts"]:
        print(f"error: no known artifacts under {args.build_dir}",
              file=sys.stderr)
        return 1

    report["target"] = args.label
    text = render(report, args.label)
    print(text, end="")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "lib_size.txt").write_text(text)
        (args.output_dir /
         "lib_size.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output_dir / "lib_size_metrics.txt").write_text(
            metrics(report, args.label))
    return 0


if __name__ == "__main__":
    sys.exit(main())
