#!/usr/bin/env python3
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
"""Compare user-facing Edge-LLM surfaces between two git revisions."""

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

SURFACE_PATHS = (
    "pyproject.toml",
    "CHANGELOG.md",
    "CMakeLists.txt",
    "cmake",
    "docs/source/user_guide/getting_started",
    "docs/source/user_guide/examples/server.md",
    "tensorrt_edgellm/cli.py",
    "tensorrt_edgellm/scripts",
    "tensorrt_edgellm/onnx/cli.py",
    "tensorrt_edgellm/builder/cli.py",
    "tensorrt_edgellm/server/cli.py",
    "tensorrt_edgellm/server/config.py",
    "tensorrt_edgellm/experimental/frontend/cli.py",
    "examples/cpp",
    "cpp/plugins",
    "cpp/runtime/config",
)


def _git(*args: str, check: bool = True) -> str:
    result = subprocess.run(("git", *args),
                            check=check,
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    return result.stdout


def _show(ref: str, path: str) -> str | None:
    result = subprocess.run(("git", "show", f"{ref}:{path}"),
                            check=False,
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    return result.stdout if result.returncode == 0 else None


def _pyproject(ref: str) -> dict[str, Any]:
    raw = _show(ref, "pyproject.toml")
    if raw is None:
        return {
            "requires_python": None,
            "dependencies": [],
            "scripts": {},
            "extras": {}
        }
    parsed = tomllib.loads(raw)
    project = parsed.get("project", {})
    return {
        "requires_python": project.get("requires-python"),
        "dependencies": project.get("dependencies", []),
        "scripts": project.get("scripts", {}),
        "extras": {
            name: dependencies
            for name, dependencies in project.get("optional-dependencies",
                                                  {}).items()
        },
    }


def _cmake_options(ref: str) -> dict[str, str | None]:
    paths = [
        path
        for path in _git("ls-tree", "-r", "--name-only", ref).splitlines()
        if (path.endswith(".cmake") or path.endswith("CMakeLists.txt"))
        and not path.startswith("3rdParty/")
    ]
    options: dict[str, str | None] = {}
    # option(NAME "doc" DEFAULT) and set(NAME VALUE CACHE TYPE "doc")
    option_pattern = re.compile(
        r"\boption\s*\(\s*([A-Za-z0-9_]+)(?:\s+\"[^\"]*\")?(?:\s+([A-Za-z0-9_]+))?",
        re.IGNORECASE,
    )
    cache_pattern = re.compile(
        r"\bset\s*\(\s*([A-Za-z0-9_]+)\s+([^\s)]+)\s+CACHE\b",
        re.IGNORECASE,
    )
    for path in paths:
        raw = _show(ref, path)
        if raw is None:
            continue
        for match in option_pattern.finditer(raw):
            options[match.group(1)] = match.group(2)
        for match in cache_pattern.finditer(raw):
            options[match.group(1)] = match.group(2)
    return dict(sorted(options.items()))


def _mapping(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_keys = set(before)
    after_keys = set(after)
    return {
        "added": sorted(after_keys - before_keys),
        "removed": sorted(before_keys - after_keys),
        "changed": {
            key: {
                "from": before[key],
                "to": after[key]
            }
            for key in sorted(before_keys & after_keys)
            if before[key] != after[key]
        },
    }


def compare(from_ref: str, to_ref: str) -> dict[str, Any]:
    _git("rev-parse", "--verify", f"{from_ref}^{{commit}}")
    _git("rev-parse", "--verify", f"{to_ref}^{{commit}}")
    before_project = _pyproject(from_ref)
    after_project = _pyproject(to_ref)
    changed = _git("diff", "--name-status", from_ref, to_ref, "--",
                   *SURFACE_PATHS).splitlines()
    return {
        "from":
        _git("rev-parse", from_ref).strip(),
        "to":
        _git("rev-parse", to_ref).strip(),
        "requires_python": {
            "from": before_project["requires_python"],
            "to": after_project["requires_python"],
        },
        "dependencies":
        _mapping({d: d
                  for d in before_project["dependencies"]},
                 {d: d
                  for d in after_project["dependencies"]}),
        "commands":
        _mapping(before_project["scripts"], after_project["scripts"]),
        "extras":
        _mapping(before_project["extras"], after_project["extras"]),
        "cmake_options":
        _mapping(_cmake_options(from_ref), _cmake_options(to_ref)),
        "changed_surface_paths":
        changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Edge-LLM CLI, package, CMake, and docs surfaces")
    parser.add_argument("from_ref")
    parser.add_argument("to_ref")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.from_ref, args.to_ref)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
