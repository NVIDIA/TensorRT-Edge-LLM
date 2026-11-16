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
"""Load CMake build metadata and bind artifact checks to configured evidence."""
# yapf: disable

import dataclasses
import hashlib
import json
import os
import pathlib
import shlex
import typing

import common


@dataclasses.dataclass(frozen=True)
class CodemodelSelection:
    """Selected CMake ``generator``, canonical ``source_root``, and target references."""
    generator: str
    source_root: pathlib.Path
    target_references: typing.Tuple[typing.Mapping[str, typing.Any], ...]


@dataclasses.dataclass(frozen=True)
class SourceRecord:
    """Configured canonical ``path`` and optional compilation group index."""
    path: pathlib.Path
    compile_group_index: typing.Optional[int]


@dataclasses.dataclass(frozen=True)
class CMakeTarget:
    """Normalized ``id``/``name``/``type``, sources/artifacts, dependency IDs, link fragments, and build directory."""
    id: str
    name: str
    type: str
    sources: typing.Tuple[SourceRecord, ...]
    artifacts: typing.Tuple[pathlib.Path, ...]
    dependency_ids: typing.Tuple[str, ...]
    link_fragments: typing.Tuple[str, ...]
    build_directory: pathlib.Path


@dataclasses.dataclass(frozen=True)
class CMakeSnapshot:
    """Configured ``configuration``/``generator`` with ``targets``, ``compile_outputs``, and exact ``tool_paths``."""
    configuration: str
    generator: str
    targets: typing.Tuple[CMakeTarget, ...]
    compile_outputs: typing.Mapping[typing.Tuple[str, pathlib.Path], pathlib.Path]
    tool_paths: typing.Mapping[str, pathlib.Path]


def load_snapshot(repo_root: pathlib.Path, build_root: pathlib.Path, configuration: str,
                  compile_target_names: typing.AbstractSet[str]) -> CMakeSnapshot:
    """Return normalized evidence for the roots/configuration; raise ``BoundaryError`` on inconsistency."""
    model = select_codemodel(build_root, configuration)
    if model.source_root != repo_root:
        raise common.BoundaryError("CMake source root does not match repo-root")
    targets = load_targets(build_root, model.source_root, model.target_references)
    compile_targets = tuple(target for target in targets if target.name in compile_target_names)
    compile_outputs = load_compile_outputs(build_root, compile_targets)
    tools = load_tool_paths(build_root)
    known_ids = {target.id for target in targets}
    for target in targets:
        if any(dependency not in known_ids for dependency in target.dependency_ids):
            raise common.BoundaryError(f"target dependency ID is absent: {target.name}")
    return CMakeSnapshot(configuration, model.generator, targets, compile_outputs, tools)


def select_codemodel(build_root: pathlib.Path, configuration: str) -> CodemodelSelection:
    """Args: build root/configuration. Returns: selection. Raises: BoundaryError for invalid File API evidence."""
    reply_root, index = select_index(build_root)
    codemodel = load_codemodel_reply(reply_root, index)
    selected = select_configuration(codemodel, configuration)
    try:
        generator = index["cmake"]["generator"]["name"]
        source_root = pathlib.Path(codemodel["paths"]["source"]).resolve(strict=True)
        reply_build_root = pathlib.Path(codemodel["paths"]["build"]).resolve(strict=True)
        references = tuple(selected["targets"])
    except (KeyError, TypeError, OSError) as error:
        raise common.BoundaryError(f"codemodel selection is malformed: {error}") from error
    if reply_build_root != pathlib.Path(build_root).resolve(strict=True):
        raise common.BoundaryError("CMake build root does not match build-dir")
    return CodemodelSelection(generator, source_root, references)


def select_index(build_root: pathlib.Path) -> typing.Tuple[pathlib.Path, typing.Mapping[str, typing.Any]]:
    """Args: build root. Returns: reply root/index. Raises: BoundaryError when no bounded index exists."""
    reply_root = pathlib.Path(build_root) / ".cmake/api/v1/reply"
    indexes = sorted(reply_root.glob("index-*.json"), key=lambda path: path.name)
    if not indexes:
        raise common.BoundaryError("CMake File API index is missing")
    return reply_root, _read_json(indexes[-1], reply_root, 1048576)


def load_codemodel_reply(reply_root: pathlib.Path,
                         index: typing.Mapping[str, typing.Any]) -> typing.Mapping[str, typing.Any]:
    """Args: reply root/index. Returns: codemodel. Raises: BoundaryError unless exactly one v2 reply exists."""
    try:
        client = index["reply"]["client-edgellm-safety"]
        responses = client["query.json"]["responses"] if "query.json" in client else list(client.values())
        matches = [row for row in responses if isinstance(row, dict) and row.get("kind") == "codemodel"
                   and row.get("version", {}).get("major") == 2 and isinstance(row.get("jsonFile"), str)]
    except (KeyError, TypeError) as error:
        raise common.BoundaryError(f"CMake client reply is malformed: {error}") from error
    if len(matches) != 1:
        raise common.BoundaryError("exactly one codemodel version-two response is required")
    return _read_json(reply_root / matches[0]["jsonFile"], reply_root, 4194304)


def select_configuration(codemodel: typing.Mapping[str, typing.Any],
                         configuration: str) -> typing.Mapping[str, typing.Any]:
    """Args: codemodel/name. Returns: row. Raises: BoundaryError unless exactly one row matches."""
    rows = codemodel.get("configurations")
    if not isinstance(rows, list):
        raise common.BoundaryError("codemodel configurations must be a list")
    matches = [row for row in rows if isinstance(row, dict) and row.get("name") == configuration]
    if len(matches) != 1:
        raise common.BoundaryError(f"exact CMake configuration not found: {configuration}")
    return matches[0]


def load_targets(build_root: pathlib.Path, source_root: pathlib.Path,
                 references: typing.Sequence[typing.Mapping[str, typing.Any]]) -> typing.Tuple[CMakeTarget, ...]:
    """Args: roots/references. Returns: targets. Raises: BoundaryError for invalid or duplicate targets."""
    reply_root = pathlib.Path(build_root) / ".cmake/api/v1/reply"
    targets = tuple(load_target(reply_root, build_root, source_root, reference) for reference in references)
    ids = [target.id for target in targets]
    names = [target.name for target in targets]
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        raise common.BoundaryError("CMake target IDs and names must be unique")
    return tuple(sorted(targets, key=lambda target: target.name))


def load_target(reply_root: pathlib.Path, build_root: pathlib.Path, source_root: pathlib.Path,
                reference: typing.Mapping[str, typing.Any]) -> CMakeTarget:
    """Args: roots/reference. Returns: target. Raises: BoundaryError for malformed identity or paths."""
    try:
        document = _read_json(reply_root / reference["jsonFile"], reply_root, 4194304)
        if document["id"] != reference["id"] or document["name"] != reference["name"]:
            raise common.BoundaryError("CMake target reference identity changed")
        sources = load_sources(document, build_root, source_root)
        artifacts = tuple(_rooted_artifact(build_root, row["path"]) for row in document.get("artifacts", []))
        build_directory = _rooted(build_root, document["paths"]["build"])
        dependencies = tuple(row["id"] for row in document.get("dependencies", []))
        fragments = tuple(row["fragment"] for row in document.get("link", {}).get("commandFragments", []))
        return CMakeTarget(document["id"], document["name"], document["type"], sources, artifacts, dependencies,
                           fragments, build_directory)
    except common.BoundaryError:
        raise
    except (KeyError, TypeError, OSError) as error:
        raise common.BoundaryError(f"CMake target reply is malformed: {error}") from error


def load_sources(document: typing.Mapping[str, typing.Any], build_root: pathlib.Path,
                 source_root: pathlib.Path) -> typing.Tuple[SourceRecord, ...]:
    """Args: target/roots. Returns: sources. Raises: BoundaryError for malformed groups or escaped paths."""
    groups = document.get("compileGroups", [])
    rows = document.get("sources", [])
    if not isinstance(groups, list) or not isinstance(rows, list):
        raise common.BoundaryError("target source records must be lists")
    sources = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise common.BoundaryError("target source record is malformed")
        group = row.get("compileGroupIndex")
        if group is not None and (type(group) is not int or group < 0 or group >= len(groups)):
            raise common.BoundaryError(
                f"source compileGroupIndex is invalid: {row['path']}")
        root = build_root if row.get("isGenerated", False) else source_root
        sources.append(SourceRecord(_rooted(root, row["path"]), group))
    return tuple(sorted(sources, key=lambda source: str(source.path)))

def load_compile_outputs(build_root: pathlib.Path, targets: typing.Sequence[CMakeTarget]) \
        -> typing.Mapping[typing.Tuple[str, pathlib.Path], pathlib.Path]:
    """Args: build root/targets. Returns: object map. Raises: BoundaryError for ambiguous provenance."""
    rows = _read_json(pathlib.Path(build_root) / "compile_commands.json", build_root, 16777216)
    if not isinstance(rows, list):
        raise common.BoundaryError("compile_commands.json must contain a list")
    targets_by_name = {target.name: target for target in targets}
    outputs = {}
    observed_objects = set()
    for row in rows:
        try:
            directory = pathlib.Path(row["directory"]).resolve(strict=False)
            if not common.is_below(directory, build_root):
                raise common.BoundaryError(
                    f"compile directory escapes build root: {directory}")
            source = _resolve_from(directory, row["file"])
            object_value = row.get("output")
            object_path = (_rooted(build_root, object_value) if object_value
                           else _rooted(directory, _output_from_command(row)))
        except (KeyError, TypeError) as error:
            raise common.BoundaryError(
                f"compile command row is malformed: {error}") from error
        owner = _compile_owner(object_path)
        if owner not in targets_by_name:
            continue
        owner_root = targets_by_name[owner].build_directory / "CMakeFiles" / (owner + ".dir")
        compiled = {record.path for record in targets_by_name[owner].sources if record.compile_group_index is not None}
        key = (owner, source)
        if (source not in compiled or not common.is_below(object_path, owner_root)
                or key in outputs or object_path in observed_objects):
            raise common.BoundaryError(
                f"invalid target-owned compile output: {object_path}")
        outputs[key] = object_path
        observed_objects.add(object_path)
    for target in targets:
        for source in target.sources:
            if source.compile_group_index is not None and (target.name, source.path) not in outputs:
                raise common.BoundaryError(
                    f"configured source has no target-owned object: {source.path}")
    return dict(sorted(outputs.items(), key=lambda item: (item[0][0], str(item[0][1]))))


def load_dependency_paths(
        options: common.Options,
        object_path: pathlib.Path) -> typing.Tuple[pathlib.Path, ...]:
    """Return repository paths named by one compiler-generated object depfile."""
    depfile = pathlib.Path(f"{object_path}.d")
    tokens = _load_dependency_tokens(depfile, options.build_root)
    paths = (_resolve_repository_dependency(token, options)
             for token in tokens)
    return tuple(sorted({path for path in paths if path is not None}, key=str))


def _load_dependency_tokens(depfile: pathlib.Path,
                            build_root: pathlib.Path) -> typing.List[str]:
    """Return dependency tokens from one bounded compiler-generated depfile."""
    text = common.read_text(depfile, build_root, 16777216)
    logical = text.replace("\\\n", " ")
    if ":" not in logical:
        raise common.BoundaryError(
            f"compiler dependency evidence is malformed: {depfile}")
    try:
        tokens = shlex.split(logical.split(":", 1)[1], posix=True)
    except ValueError as error:
        raise common.BoundaryError(
            f"compiler dependency evidence is malformed: {depfile}: {error}"
        ) from error
    if not tokens:
        raise common.BoundaryError(
            f"compiler dependency evidence is empty: {depfile}")
    return tokens


def _resolve_repository_dependency(
        token: str, options: common.Options) -> typing.Optional[pathlib.Path]:
    """Return one canonical repository dependency, or None for external paths."""
    candidate = pathlib.Path(token)
    if not candidate.is_absolute():
        candidate = options.build_root / candidate
    canonical = candidate.resolve(strict=False)
    return canonical if common.is_below(canonical, options.repo_root) else None


def load_tool_paths(build_root: pathlib.Path) -> typing.Mapping[str, pathlib.Path]:
    """Args: build root. Returns: configured tools. Raises: BoundaryError for absent/non-file tools."""
    lines = common.read_text(
        pathlib.Path(build_root) / "CMakeCache.txt", build_root, 8388608).splitlines()
    tools = {}
    for name in ("CMAKE_AR", "CMAKE_NM", "CMAKE_READELF"):
        matches = [line.split("=", 1)[1] for line in lines if line.startswith(name + ":") and "=" in line]
        if len(matches) != 1:
            raise common.BoundaryError(
                f"required CMake tool entry is missing or duplicated: {name}")
        raw = pathlib.Path(matches[0])
        if not raw.is_absolute():
            raise common.BoundaryError(f"configured CMake tool is not absolute: {raw}")
        tool = raw.resolve(strict=False)
        if not tool.is_file():
            raise common.BoundaryError(f"configured CMake tool is unavailable: {tool}")
        tools[name] = tool
    return tools


def _read_json(path: pathlib.Path, root: pathlib.Path, maximum_bytes: int) -> typing.Any:
    """Read one bounded JSON value."""
    try:
        return json.loads(common.read_text(path, root, maximum_bytes))
    except json.JSONDecodeError as error:
        raise common.BoundaryError(f"JSON evidence is malformed: {path}") from error


def _rooted(root: pathlib.Path, value: str) -> pathlib.Path:
    """Resolve a path under its declared root and enforce containment."""
    path = pathlib.Path(value)
    resolved = path.resolve(strict=False) if path.is_absolute() else (pathlib.Path(root) / path).resolve(strict=False)
    if not common.is_below(resolved, pathlib.Path(root)):
        raise common.BoundaryError(f"generated path escapes its declared root: {value}")
    return resolved


def _rooted_artifact(root: pathlib.Path, value: str) -> pathlib.Path:
    """Preserve the declared artifact leaf while resolving its parent and enforcing containment."""
    root_path = pathlib.Path(root).resolve(strict=False)
    path = pathlib.Path(value)
    candidate = path if path.is_absolute() else root_path / path
    declared = candidate.parent.resolve(strict=False) / candidate.name
    if not common.is_below(declared, root_path):
        raise common.BoundaryError(f"generated path escapes its declared root: {value}")
    if declared.exists() and not common.is_below(declared.resolve(strict=True), root_path):
        raise common.BoundaryError(f"artifact symlink escapes its declared root: {value}")
    return declared


def _resolve_from(root: pathlib.Path, value: str) -> pathlib.Path:
    """Resolve a possibly relative path without imposing containment."""
    path = pathlib.Path(value)
    return path.resolve(strict=False) if path.is_absolute() else (root / path).resolve(strict=False)


def _output_from_command(row: typing.Mapping[str, typing.Any]) -> str:
    """Extract the value following the single compile ``-o`` option."""
    tokens = list(row.get("arguments", [])) or shlex.split(row.get("command", ""))
    indexes = [index for index, token in enumerate(tokens) if token == "-o"]
    if len(indexes) != 1 or indexes[0] + 1 >= len(tokens):
        raise common.BoundaryError("compile command has no unique output")
    return tokens[indexes[0] + 1]


def _compile_owner(object_path: pathlib.Path) -> typing.Optional[str]:
    """Return a CMake target name encoded in an object path."""
    parts = object_path.parts
    for index, part in enumerate(parts[:-1]):
        if part == "CMakeFiles" and parts[index + 1].endswith(".dir"):
            return parts[index + 1][:-4]
    return None


def check_plan_drift(context: common.BoundaryContext) -> typing.List[common.Finding]:
    """Return a finding when current build metadata differs from the configured plan."""
    if context.configured_plan is None:
        raise common.BoundaryError("artifact phase requires configured plan")
    actual = canonicalize(context)
    if context.configured_plan == actual:
        return []
    keys = sorted(set(context.configured_plan) | set(actual))
    changed = [key for key in keys if context.configured_plan.get(key) != actual.get(key)]
    return [common.Finding("plan_drift", kind="changed", expected="configured snapshot", actual=",".join(changed))]


def canonicalize(context: common.BoundaryContext) -> typing.Mapping[str, typing.Any]:
    """Return the canonical policy, target, source, object, link, and tool evidence."""
    rows = [_target_row(context, target) for target in _select_plan_targets(context)]
    tools = {
        name: _stable_path(path, context.options.build_root)
        for name, path in sorted(context.cmake_snapshot.tool_paths.items())
    }
    return {
        "version": 1,
        "platform": context.options.platform,
        "configuration": context.options.configuration,
        "generator": context.cmake_snapshot.generator,
        "policy_sha256": context.policy.digest,
        "targets": rows,
        "tools": tools
    }


def _select_plan_targets(context: common.BoundaryContext) -> typing.List[CMakeTarget]:
    """Return the deterministic union of present policy-root dependency closures."""
    targets_by_name = {target.name: target for target in context.cmake_snapshot.targets}
    targets_by_id = {target.id: target for target in context.cmake_snapshot.targets}
    pending = [
        targets_by_name[row.name].id for row in context.policy.targets
        if row.name in targets_by_name
    ]
    selected = set()
    while pending:
        target = targets_by_id[pending.pop(0)]
        if target.id in selected:
            continue
        selected.add(target.id)
        pending.extend(target.dependency_ids)
    return sorted((targets_by_id[target_id] for target_id in selected), key=lambda target: target.name)


def _source_rows(context: common.BoundaryContext,
                 target: CMakeTarget) -> typing.List[typing.Mapping[str, typing.Any]]:
    """Return source identities and hashes for one configured target."""
    rows = []
    for source in sorted(target.sources, key=lambda record: str(record.path)):
        text = common.read_text(source.path, context.options.repo_root, 4194304)
        rows.append({
            "path": source.path.relative_to(context.options.repo_root).as_posix(),
            "compile_group_index": source.compile_group_index,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()
        })
    return rows


def _target_row(context: common.BoundaryContext, target: CMakeTarget) -> typing.Mapping[str, typing.Any]:
    """Return normalized plan evidence for one target."""
    objects = [{
        "source": source.relative_to(context.options.repo_root).as_posix(),
        "object": path.relative_to(context.options.build_root).as_posix()
    } for (owner, source), path in context.cmake_snapshot.compile_outputs.items() if owner == target.name]
    return {
        "name": target.name,
        "type": target.type,
        "dependencies": sorted(target.dependency_ids),
        "artifacts": sorted(path.relative_to(context.options.build_root).as_posix() for path in target.artifacts),
        "sources": _source_rows(context, target),
        "objects": sorted(objects, key=lambda row: (row["source"], row["object"])),
        "link_fragments": list(target.link_fragments)
    }


def write_configured_plan(context: common.BoundaryContext) -> None:
    """Atomically publish the configured plan after configured checks pass."""
    data = json.dumps(canonicalize(context), sort_keys=True, separators=(",", ":")) + "\n"
    plan_path = context.options.plan_path
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = plan_path.with_name(plan_path.name + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    os.replace(str(temporary), str(plan_path))


def load_configured_plan(options: common.Options) -> typing.Optional[typing.Mapping[str, typing.Any]]:
    """Load the configured plan for artifact checks, or return None during configuration."""
    if options.phase == "configured":
        return None
    try:
        document = json.loads(common.read_text(options.plan_path, options.build_root, 8388608))
    except json.JSONDecodeError as error:
        raise common.BoundaryError("configured plan is malformed") from error
    if not isinstance(document, dict) or document.get("version") != 1:
        raise common.BoundaryError("configured plan version 1 is required")
    return document


def _stable_path(path: pathlib.Path, build_root: pathlib.Path) -> str:
    """Return a build-relative path when possible, otherwise an absolute path."""
    try:
        return path.relative_to(build_root).as_posix()
    except ValueError:
        return str(path)
