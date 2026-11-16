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
"""Check configured target closures, source paths, headers, and prohibited APIs."""

import dataclasses
import pathlib
import typing

import build_metadata
import common

MAX_SOURCE_EVIDENCE_BYTES = 16777216


@dataclasses.dataclass(frozen=True)
class TargetExpectation:
    """Required root identity, CMake type, and exact target closure allowset."""

    name: str
    required: bool
    expected_type: str
    admitted_targets: typing.FrozenSet[str]


@dataclasses.dataclass(frozen=True)
class TargetObservation:
    """Observed target identity and outputs."""

    name: str
    type: str
    artifacts: typing.Tuple[pathlib.Path, ...]


@dataclasses.dataclass(frozen=True)
class SourceExpectation:
    """Expected source contract.
    Attributes: ``roots`` are owned trees; ``reused`` is an exact path allowset; ``prohibited_rules`` reject APIs.
    """

    roots: typing.Tuple[pathlib.Path, ...]
    reused: typing.FrozenSet[pathlib.Path]
    prohibited_rules: typing.Tuple[common.Rule, ...]


@dataclasses.dataclass(frozen=True)
class SourceObservation:
    """Observed source.
    Attributes: ``root_name``/``target_name`` identify ownership; ``path`` is display text; ``canonical_path`` is
    identity; ``text`` contains bounded repository content or None for an external path.
    """

    root_name: str
    target_name: str
    path: str
    canonical_path: pathlib.Path
    text: typing.Optional[str]


def check_target_graph(
        context: common.BoundaryContext) -> typing.List[common.Finding]:
    """Return missing, wrong-kind, and unadmitted-target findings."""
    expected = [
        TargetExpectation(row.name, row.required, row.cmake_type,
                          row.admitted_targets)
        for row in context.policy.targets
    ]
    targets_by_name = {
        target.name: target
        for target in context.cmake_snapshot.targets
    }
    targets_by_id = {
        target.id: target
        for target in context.cmake_snapshot.targets
    }
    actual = {}
    for policy in context.policy.targets:
        if policy.name not in targets_by_name:
            actual[policy.name] = []
            continue
        pending = [targets_by_name[policy.name].id]
        visited = set()
        closure = []
        while pending:
            target_id = pending.pop(0)
            if target_id in visited:
                continue
            target = targets_by_id[target_id]
            visited.add(target_id)
            closure.append(
                TargetObservation(target.name, target.type, target.artifacts))
            pending.extend(target.dependency_ids)
        actual[policy.name] = sorted(closure, key=lambda row: row.name)
    return compare_target_graph(sorted(expected, key=lambda row: row.name),
                                actual)


def compare_target_graph(
    expected: typing.Sequence[TargetExpectation],
    actual: typing.Mapping[str, typing.Sequence[TargetObservation]]
) -> typing.List[common.Finding]:
    """Compare required root closures with target policy."""
    findings = []
    for root in expected:
        closure = actual[root.name]
        if not closure:
            if root.required:
                findings.append(
                    common.Finding("target_graph", root.name, "missing_root"))
            continue
        root_actual = next(row for row in closure if row.name == root.name)
        if root_actual.type != root.expected_type:
            findings.append(
                common.Finding("target_graph", root.name, "target_type",
                               root.expected_type, root_actual.type))
        if len(root_actual.artifacts) != 1:
            findings.append(
                common.Finding("target_graph", root.name, "artifact_count",
                               "1", str(len(root_actual.artifacts))))
        for target in closure:
            if target.name not in root.admitted_targets:
                findings.append(
                    common.Finding("target_graph",
                                   root.name,
                                   "unadmitted_target",
                                   actual=target.name))
    return sorted(findings)


def check_sources(
        context: common.BoundaryContext) -> typing.List[common.Finding]:
    """Return unadmitted-path and prohibited-API findings for configured and compiled sources."""
    expected = source_expectation(context)
    rows = collect_root_sources(context)
    if context.options.phase == "artifacts":
        rows.extend(collect_compiled_dependencies(context))
    actual = [
        load_source_observation(context, row)
        for row in sorted(set(rows),
                          key=lambda row: (row[0], row[1], str(row[2])))
    ]
    return compare_sources(context, expected, actual)


def source_expectation(context: common.BoundaryContext) -> SourceExpectation:
    """Return the source path and prohibited-API contract."""
    repo_root = context.options.repo_root
    roots = (repo_root / "cpp/safety_inference",
             repo_root / "cpp/safety_compatible",
             repo_root / "examples/safety", repo_root / "tests/safety")
    reused = frozenset((repo_root / path).resolve(strict=False)
                       for path in context.policy.reused_sources)
    reused |= load_configured_generated_sources(context)
    validate_rule_paths(repo_root, context.policy.source_rules)
    return SourceExpectation(roots, reused, context.policy.source_rules)


def compare_sources(
        context: common.BoundaryContext, expected: SourceExpectation,
        actual: typing.Sequence[SourceObservation]
) -> typing.List[common.Finding]:
    """Compare observed source paths and contents with the source contract."""
    findings = []
    for source in actual:
        under_root = any(
            common.is_below(source.canonical_path, root)
            for root in expected.roots)
        if not under_root and source.canonical_path not in expected.reused:
            findings.append(
                common.Finding("source",
                               source.root_name,
                               "unadmitted_path",
                               actual=source.path,
                               origin=source.target_name))
        if source.text is None:
            continue
        for rule in expected.prohibited_rules:
            if not rule_applies(rule, source.canonical_path,
                                context.options.repo_root):
                continue
            for line_number, line in enumerate(source.text.splitlines(),
                                               start=1):
                if rule.expression.search(line) is not None:
                    findings.append(
                        common.Finding("source",
                                       source.root_name,
                                       rule.id,
                                       actual=source.path,
                                       origin=f"line {line_number}"))
    return sorted(findings)


def validate_rule_paths(repo_root: pathlib.Path,
                        rules: typing.Sequence[common.Rule]) -> None:
    """Reject scoped rules that name absolute paths or escape the repository."""
    for rule in rules:
        for value in rule.paths:
            path = pathlib.Path(value)
            if path.is_absolute() or not common.is_below(
                (repo_root / path).resolve(strict=False), repo_root):
                raise common.BoundaryError(
                    f"source rule path escapes the repository: {value}")


def rule_applies(rule: common.Rule, source: pathlib.Path,
                 repo_root: pathlib.Path) -> bool:
    """Return whether an unscoped rule or one of its repository paths owns the source."""
    return not rule.paths or any(
        common.is_below(source, (repo_root / path).resolve(strict=False))
        for path in rule.paths)


def collect_root_sources(context: common.BoundaryContext) \
        -> typing.List[typing.Tuple[str, str, pathlib.Path]]:
    """Args: shared context. Returns: unique root, owner, and source tuples from present closures."""
    targets_by_name = {
        target.name: target
        for target in context.cmake_snapshot.targets
    }
    targets_by_id = {
        target.id: target
        for target in context.cmake_snapshot.targets
    }
    observed = set()
    for policy in context.policy.targets:
        if policy.name not in targets_by_name:
            continue
        pending = [targets_by_name[policy.name].id]
        visited = set()
        while pending:
            target = targets_by_id[pending.pop(0)]
            if target.id in visited:
                continue
            visited.add(target.id)
            observed.update((policy.name, target.name, source.path)
                            for source in target.sources)
            pending.extend(target.dependency_ids)
    return sorted(observed, key=lambda row: (row[0], row[1], str(row[2])))


def collect_compiled_dependencies(context: common.BoundaryContext) \
        -> typing.List[typing.Tuple[str, str, pathlib.Path]]:
    """Return repository dependencies from each root target's compiled objects."""
    root_owners = {
        (root, owner)
        for root, owner, unused_path in collect_root_sources(context)
    }
    observed = set()
    for root_name, owner in root_owners:
        for (compiled_owner, unused_source
             ), object_path in context.cmake_snapshot.compile_outputs.items():
            if compiled_owner != owner:
                continue
            observed.update((root_name, owner, path)
                            for path in build_metadata.load_dependency_paths(
                                context.options, object_path))
    return sorted(observed, key=lambda row: (row[0], row[1], str(row[2])))


def load_source_observation(
        context: common.BoundaryContext,
        row: typing.Tuple[str, str, pathlib.Path]) -> SourceObservation:
    """Args: context and source row. Returns: observation. Raises: BoundaryError for invalid source evidence."""
    root_name, target_name, source = row
    canonical = source.resolve(strict=False)
    inside_repo = common.is_below(canonical, context.options.repo_root)
    text = common.read_text(canonical, context.options.repo_root,
                            MAX_SOURCE_EVIDENCE_BYTES) if inside_repo else None
    display = canonical.relative_to(context.options.repo_root).as_posix(
    ) if inside_repo else str(canonical)
    return SourceObservation(root_name, target_name, display, canonical, text)


def load_configured_generated_sources(
        context: common.BoundaryContext) -> typing.FrozenSet[pathlib.Path]:
    """Return exact generated source paths declared by the configured build."""
    cache_path = context.options.build_root / "CMakeCache.txt"
    if not cache_path.is_file():
        return frozenset()
    cache = common.read_text(cache_path, context.options.build_root,
                             8388608).splitlines()
    prefix = "EDGELLM_SAFETY_FMHA_ARTIFACT_DIR:"
    matches = [
        line.split("=", 1)[1] for line in cache
        if line.startswith(prefix) and "=" in line
    ]
    if not matches:
        return frozenset()
    if len(matches) != 1:
        raise common.BoundaryError(
            "Safety FMHA artifact directory is duplicated in CMakeCache.txt")
    artifact_root = pathlib.Path(matches[0])
    if not artifact_root.is_absolute():
        raise common.BoundaryError(
            "Safety FMHA artifact directory is not absolute")
    return frozenset({
        (artifact_root / "safetyFmhaArtifacts.h").resolve(strict=False)
    })
