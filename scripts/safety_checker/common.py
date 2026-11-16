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
"""Shared records and evidence primitives for the Safety Runtime boundary gate."""
# yapf: disable

import dataclasses
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import typing

import yaml

SUPPORTED_PLATFORMS = ("x86", "d7l", "d7qs")
ARTIFACT_TYPES = {"static_archive": "STATIC_LIBRARY", "shared_library": "SHARED_LIBRARY", "executable": "EXECUTABLE"}


class BoundaryError(ValueError):
    """Indicate malformed, missing, or unsafe boundary evidence."""


@dataclasses.dataclass(frozen=True)
class Options:
    """Canonical ``repo_root``/``build_root``, ``platform``/``configuration``/``phase``, and plan/report paths."""
    repo_root: pathlib.Path
    build_root: pathlib.Path
    platform: str
    configuration: str
    phase: str
    plan_path: pathlib.Path
    report_path: pathlib.Path


@dataclasses.dataclass(frozen=True)
class Rule:
    """Policy rule with stable ``id``/``pattern`` and optional source ``paths`` or artifact ``targets``."""
    id: str
    pattern: str
    expression: typing.Pattern[str] = dataclasses.field(compare=False, repr=False)
    paths: typing.Tuple[str, ...] = ()
    targets: typing.Tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class TargetPolicy:
    """Root ``name``/``required``, ``admitted_targets``, and expected ``artifact_kind``/``cmake_type``."""
    name: str
    required: bool
    admitted_targets: typing.FrozenSet[str]
    artifact_kind: str
    cmake_type: str


@dataclasses.dataclass(frozen=True)
class BoundaryPolicy:
    """Selected ``targets``, ``reused_sources``, ``source_rules``, ``symbol_rules``, ``link_rules``, and
    ``forbidden_link_rules`` plus ``dynamic_names``, ``digest``, and ``runtime_consumed``."""
    targets: typing.Tuple[TargetPolicy, ...]
    reused_sources: typing.Tuple[str, ...]
    source_rules: typing.Tuple[Rule, ...]
    symbol_rules: typing.Tuple[Rule, ...]
    link_rules: typing.Tuple[Rule, ...]
    forbidden_link_rules: typing.Tuple[Rule, ...]
    dynamic_names: typing.FrozenSet[str]
    digest: str
    runtime_consumed: bool = False


@dataclasses.dataclass(frozen=True)
class BoundaryContext:
    """Fresh options, policy, CMake inputs, and optional configured plan."""
    options: Options
    policy: BoundaryPolicy
    cmake_snapshot: typing.Any
    configured_plan: typing.Optional[typing.Mapping[str, typing.Any]]


@dataclasses.dataclass(frozen=True, order=True)
class Finding:
    """Difference with ``check``/``target``/``kind``, ``expected``/``actual`` values, and ``origin``."""
    check: str
    target: str = ""
    kind: str = ""
    expected: str = ""
    actual: str = ""
    origin: str = ""


def is_below(path: pathlib.Path, root: pathlib.Path) -> bool:
    """Return whether candidate ``path`` is equal to or below ``root``."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def read_text(path: pathlib.Path, required_root: pathlib.Path, maximum_bytes: int) -> str:
    """Return bounded UTF-8 ``path`` below ``required_root``; raise ``BoundaryError`` for invalid evidence."""
    try:
        return read_bytes(path, required_root, maximum_bytes).decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise BoundaryError(f"unable to decode evidence {path}: {error}") from error


def read_bytes(path: pathlib.Path, required_root: pathlib.Path, maximum_bytes: int) -> bytes:
    """Return bounded ``path`` bytes below ``required_root`` or raise ``BoundaryError``."""
    try:
        root = pathlib.Path(required_root).resolve(strict=True)
        resolved = pathlib.Path(path).resolve(strict=True)
        metadata = resolved.stat()
        if not is_below(resolved, root) or not resolved.is_file() or metadata.st_size > maximum_bytes:
            raise BoundaryError(f"evidence must be a bounded regular file below {root}: {path}")
        data = resolved.read_bytes()
        if len(data) > maximum_bytes:
            raise BoundaryError(f"evidence exceeds size limit: {path}")
        return data
    except BoundaryError:
        raise
    except OSError as error:
        raise BoundaryError(f"unable to read evidence {path}: {error}") from error


def run_bytes(executable: pathlib.Path, arguments: typing.Sequence[typing.Any], maximum_bytes: int) -> bytes:
    """Run a configured tool and return bounded raw stdout or raise ``BoundaryError``."""
    tool = pathlib.Path(executable).resolve(strict=False)
    if not tool.is_file():
        raise BoundaryError(f"configured inspection tool is missing: {executable}")
    try:
        completed = subprocess.run([str(tool)] +
                                   [str(value) for value in arguments],
                                   check=False,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BoundaryError(f"inspection tool could not run: {tool}: {error}") from error
    if len(completed.stdout) + len(completed.stderr) > maximum_bytes:
        raise BoundaryError(f"inspection tool output exceeded limit: {tool}")
    if completed.returncode != 0:
        stderr = completed.stderr[:4096].decode("utf-8", errors="replace")
        raise BoundaryError(f"inspection tool failed: {tool}: {stderr}")
    return completed.stdout


def run_lines(executable: pathlib.Path, arguments: typing.Sequence[typing.Any], maximum_bytes: int) -> typing.List[str]:
    """Run configured ``executable`` with ``arguments`` and return bounded UTF-8 lines or raise ``BoundaryError``."""
    data = run_bytes(executable, arguments, maximum_bytes)
    try:
        return data.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as error:
        raise BoundaryError(f"inspection tool output is not UTF-8: {executable}") from error


def load_policy(repo_root: pathlib.Path, platform: str) -> BoundaryPolicy:
    """Return the selected policy for ``repo_root``/``platform`` or raise ``BoundaryError`` for invalid policy."""
    document, gate = load_policy_document(repo_root)
    runtime_consumed = document.get("runtime_consumed")
    if type(runtime_consumed) is not bool:
        raise BoundaryError("runtime_consumed must be Boolean")
    targets = load_target_policies(gate.get("targets"), runtime_consumed)
    profile = select_platform_profile(gate.get("platforms"), platform)
    reused = _string_list(document.get("reused_standard_sources"), "reused_standard_sources")
    selected = {
        "version": 1,
        "runtime_consumed": runtime_consumed,
        "targets": gate["targets"],
        "reused_sources": reused,
        "source_rules": gate.get("source_rules"),
        "undefined_symbol_rules": gate.get("undefined_symbol_rules"),
        "admitted_link_patterns": profile["admitted_link_patterns"],
        "forbidden_link_patterns": profile["forbidden_link_patterns"],
        "admitted_dynamic_libraries": profile["admitted_dynamic_libraries"]}
    digest = hashlib.sha256(json.dumps(selected, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    dynamic = _string_list(profile["admitted_dynamic_libraries"], "admitted_dynamic_libraries")
    if len(dynamic) != len(set(dynamic)):
        raise BoundaryError("dynamic library names must be unique")
    return BoundaryPolicy(targets, tuple(reused), compile_rules(gate.get("source_rules"), "source_rules"),
                          compile_rules(gate.get("undefined_symbol_rules"), "undefined_symbol_rules",
                                        frozenset(target.name for target in targets)),
                          compile_rules(profile["admitted_link_patterns"], "admitted_link_patterns"),
                          compile_rules(profile["forbidden_link_patterns"], "forbidden_link_patterns"),
                          frozenset(dynamic), digest, runtime_consumed)


def load_policy_document(repo_root: pathlib.Path) -> typing.Tuple[typing.Mapping[str, typing.Any],
                                                                  typing.Mapping[str, typing.Any]]:
    """Args: repository root. Returns: document/gate. Raises: BoundaryError for invalid YAML or version."""
    text = read_text(pathlib.Path(repo_root) / "cpp/safety_inference/support_index.yaml", repo_root, 262144)
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise BoundaryError(f"support index YAML is invalid: {error}") from error
    if not isinstance(document, dict) or not isinstance(document.get("boundary_gate"), dict):
        raise BoundaryError("boundary_gate mapping is required")
    gate = document["boundary_gate"]
    if gate.get("version") != 1:
        raise BoundaryError("boundary_gate version 1 is required")
    return document, gate


def load_target_policies(rows: typing.Any, runtime_consumed: bool = False) -> typing.Tuple[TargetPolicy, ...]:
    """Args: rows/consumption state. Returns: policies. Raises: BoundaryError for invalid rows."""
    if not isinstance(rows, dict) or not rows:
        raise BoundaryError("boundary targets mapping is required")
    targets = []
    for name, row in rows.items():
        if not isinstance(name, str) or not isinstance(row, dict) or type(row.get("required")) is not bool:
            raise BoundaryError("boundary target rows are malformed")
        admitted = _string_list(row.get("admitted_targets"), f"{name}.admitted_targets")
        kind = row.get("artifact_kind")
        if name not in admitted or len(admitted) != len(set(admitted)) or kind not in ARTIFACT_TYPES:
            raise BoundaryError(f"invalid boundary target policy: {name}")
        if not row["required"] and (name != "llm_inference_safe" or runtime_consumed):
            raise BoundaryError("only an unconsumed llm_inference_safe target may be optional")
        targets.append(TargetPolicy(name, row["required"], frozenset(admitted), kind, ARTIFACT_TYPES[kind]))
    return tuple(sorted(targets, key=lambda item: item.name))


def select_platform_profile(platforms: typing.Any, platform: str) -> typing.Mapping[str, typing.Any]:
    """Args: profiles/platform. Returns: profile. Raises: BoundaryError for absent lists."""
    if not isinstance(platforms, dict) or not isinstance(platforms.get(platform), dict):
        raise BoundaryError(f"selected platform policy is missing: {platform}")
    profile = platforms[platform]
    for field in ("admitted_link_patterns", "forbidden_link_patterns", "admitted_dynamic_libraries"):
        if not isinstance(profile.get(field), list):
            raise BoundaryError(f"platform policy field must be a list: {field}")
    return profile


def compile_rules(rows: typing.Any, field_name: str,
                  valid_targets: typing.Collection[str] = ()) -> typing.Tuple[Rule, ...]:
    """Args: rows/field name. Returns: rules. Raises: BoundaryError for duplicates or bad regex."""
    if not isinstance(rows, list):
        raise BoundaryError(f"{field_name} must be a list")
    rules = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("pattern"), str):
            raise BoundaryError(f"{field_name} contains a malformed rule")
        try:
            expression = re.compile(row["pattern"])
        except re.error as error:
            raise BoundaryError(f"{field_name} contains an invalid pattern: {error}") from error
        paths = _string_list(row.get("paths", []), f"{field_name}.paths")
        if paths and field_name != "source_rules":
            raise BoundaryError(f"{field_name} rules cannot contain paths")
        targets = _string_list(row.get("targets", []), f"{field_name}.targets")
        if targets and field_name != "undefined_symbol_rules":
            raise BoundaryError(f"{field_name} rules cannot contain targets")
        if any(target not in valid_targets for target in targets):
            raise BoundaryError(f"{field_name} contains an unknown target")
        rules.append(Rule(row["id"], row["pattern"], expression, tuple(paths), tuple(targets)))
    if len(rules) != len({rule.id for rule in rules}):
        raise BoundaryError(f"{field_name} contains duplicate rule IDs")
    return tuple(rules)


def write_report(report_path: pathlib.Path, phase: str, findings: typing.Sequence[Finding]) -> None:
    """Args: path/phase/findings. Returns: None. Raises: OSError on publication failure."""
    ordered = sorted(findings)
    document = {
        "version": 1,
        "phase": phase,
        "status": "passed" if not ordered else "failed",
        "findings": [dataclasses.asdict(finding) for finding in ordered]
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(report_path))
    for finding in ordered:
        print(f"check={finding.check} target={finding.target} kind={finding.kind} "
              f"actual={json.dumps(finding.actual)}",
              file=sys.stderr)


def _string_list(value: typing.Any, field_name: str) -> typing.List[str]:
    """Validate and return a list of nonempty strings."""
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise BoundaryError(f"{field_name} must be a list of nonempty strings")
    return list(value)
