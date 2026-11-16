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
"""Check archive contents, final links, dynamic dependencies, and symbols."""
# yapf: disable

import dataclasses
import hashlib
import pathlib
import re
import shlex
import typing

import build_metadata
import common

CONTROL_FLAGS = {"-fPIC", "-fPIE", "-pie", "-no-pie", "-shared", "-static", "-g", "-lang-c++", "-Wfatal-errors",
                 "-Wall", "-Wextra", "-Wno-unknown-pragmas", "-Wno-deprecated"}
SHELL_TOKENS = {";", "&&", "||", "|", ">", ">>", "<"}
NEEDED_PATTERN = re.compile(r".*\(NEEDED\).*Shared library:\s*\[(?P<name>[^\]\x00-\x1f]+)\]\s*")


@dataclasses.dataclass(frozen=True, order=True)
class ArchiveMember:
    """Archive member name and SHA-256 digest of its exact bytes."""

    name: str
    digest: str

@dataclasses.dataclass(frozen=True)
class LinkExpectation:
    """Exact output, owned/direct paths, project paths, and external/forbidden rules."""
    output: pathlib.Path
    own_objects: typing.FrozenSet[pathlib.Path]
    direct_artifacts: typing.FrozenSet[pathlib.Path]
    allowed_project_paths: typing.FrozenSet[pathlib.Path]
    external_patterns: typing.Tuple[common.Rule, ...]
    forbidden_external_patterns: typing.Tuple[common.Rule, ...]


@dataclasses.dataclass(frozen=True)
class LinkInput:
    """One input whose ``kind`` classifies normalized text or canonical path ``value``."""
    kind: str
    value: typing.Union[str, pathlib.Path]


@dataclasses.dataclass(frozen=True)
class LinkObservation:
    """Observed final ``output`` and structurally classified effective ``inputs``."""
    output: pathlib.Path
    inputs: typing.Tuple[LinkInput, ...]


def check_link_inputs(context: common.BoundaryContext) -> typing.List[common.Finding]:
    """Return wrong-output, missing-input, and unadmitted-input findings."""
    return compare_link_inputs(context, link_expectations(context), link_observations(context))


def link_expectations(context: common.BoundaryContext) -> typing.Mapping[str, LinkExpectation]:
    """Return final-link contracts for present shared-library and executable roots."""
    targets_by_id = {target.id: target for target in context.cmake_snapshot.targets}
    targets_by_name = {target.name: target for target in context.cmake_snapshot.targets}
    expected = {}
    for policy in context.policy.targets:
        if policy.artifact_kind not in ("shared_library", "executable"):
            continue
        root = targets_by_name.get(policy.name)
        if root is None:
            if policy.required:
                raise common.BoundaryError(f"required final-link target is absent: {policy.name}")
            continue
        if len(root.artifacts) != 1:
            raise common.BoundaryError(f"final-link target needs one artifact: {root.name}")
        own_objects = frozenset(path for (owner, unused_source), path in
                                context.cmake_snapshot.compile_outputs.items() if owner == root.name)
        direct = frozenset(path.resolve(strict=False) for target_id in root.dependency_ids
                           for path in targets_by_id[target_id].artifacts)
        dependencies = _dependency_closure(root, targets_by_id)
        closure_artifacts = frozenset(path.resolve(strict=False)
                                      for target in dependencies for path in target.artifacts)
        expected[root.name] = LinkExpectation(
            root.artifacts[0].resolve(strict=False), own_objects, direct, own_objects | closure_artifacts,
            context.policy.link_rules, context.policy.forbidden_link_rules)
    return expected


def link_observations(context: common.BoundaryContext) -> typing.Mapping[str, LinkObservation]:
    """Return parsed final-link commands for applicable configured targets."""
    if context.cmake_snapshot.generator != "Unix Makefiles":
        raise common.BoundaryError("link.txt inspection requires the Unix Makefiles generator")
    targets = {target.name: target for target in context.cmake_snapshot.targets}
    applicable = {
        row.name for row in context.policy.targets if row.artifact_kind in ("shared_library", "executable")
    }
    return {
        name: load_link_command(context.options.build_root, name, targets[name].build_directory)
        for name in sorted(applicable & set(targets))
    }


def compare_link_inputs(context: common.BoundaryContext, expected: typing.Mapping[str, LinkExpectation],
                        actual: typing.Mapping[str, LinkObservation]) -> typing.List[common.Finding]:
    """Compare observed final-link commands with configured ownership and policy."""
    findings = []
    for target_name in sorted(expected):
        wanted, observed = expected[target_name], actual[target_name]
        if observed.output != wanted.output:
            findings.append(_finding(target_name, "output", wanted.output, observed.output))
        observed_project = frozenset(item.value for item in observed.inputs if item.kind == "project_path")
        for required in sorted(wanted.own_objects | wanted.direct_artifacts, key=str):
            if required not in observed_project:
                findings.append(_finding(target_name, "missing_direct_input", required, ""))
        admitted_shared_paths = {
            item.value for item in observed.inputs
            if item.kind == "external_path" and _is_shared_library(item.value)
            and _is_admitted(str(item.value), wanted)
        }
        for item in observed.inputs:
            if item.kind == "project_path" and item.value not in wanted.allowed_project_paths:
                findings.append(_finding(target_name, "unexpected_project_input", "", item.value))
                continue
            if item.kind == "project_path":
                continue
            if item.kind == "rpath":
                admitted_parent = any(item.value == path.parent for path in admitted_shared_paths)
                if not common.is_below(item.value, context.options.build_root) and not admitted_parent:
                    findings.append(_finding(target_name, "unadmitted_rpath", "", item.value))
                continue
            value = str(item.value)
            forbidden = [
                rule.id for rule in wanted.forbidden_external_patterns
                if rule.expression.fullmatch(value) is not None
            ]
            admitted = any(rule.expression.fullmatch(value) is not None for rule in wanted.external_patterns)
            if forbidden or not admitted:
                kind = forbidden[0] if forbidden else "unadmitted_external_input"
                findings.append(_finding(target_name, kind, "", value))
    return sorted(findings)


def check_archive_contents(context: common.BoundaryContext) -> typing.List[common.Finding]:
    """Return exact missing, unexpected, and byte-content archive findings."""
    expected = {}
    for policy in context.policy.targets:
        if policy.artifact_kind != "static_archive":
            continue
        objects = [
            path for (owner, unused_source), path in context.cmake_snapshot.compile_outputs.items()
            if owner == policy.name
        ]
        names = [path.name for path in objects]
        if len(names) != len(set(names)):
            raise common.BoundaryError(f"static archive object names are ambiguous: {policy.name}")
        expected[policy.name] = tuple(
            ArchiveMember(
                path.name,
                hashlib.sha256(common.read_bytes(path, context.options.build_root, 268435456)).hexdigest())
            for path in sorted(objects, key=lambda item: item.name))

    actual = {}
    ar = context.cmake_snapshot.tool_paths["CMAKE_AR"]
    targets = {target.name: target for target in context.cmake_snapshot.targets}
    for policy in context.policy.targets:
        if policy.artifact_kind != "static_archive" or policy.name not in targets:
            continue
        target = targets[policy.name]
        if len(target.artifacts) != 1 or not target.artifacts[0].is_file():
            raise common.BoundaryError(f"static archive artifact is missing: {policy.name}")
        archive = target.artifacts[0]
        lines = [line.strip() for line in common.run_lines(ar, ["t", archive], 4194304) if line.strip()]
        if not lines:
            raise common.BoundaryError(f"static archive member evidence is empty: {policy.name}")
        if len(lines) != len(set(lines)):
            raise common.BoundaryError(f"static archive member names are ambiguous: {policy.name}")
        actual[policy.name] = tuple(
            ArchiveMember(name, hashlib.sha256(common.run_bytes(ar, ["p", archive, name], 268435456)).hexdigest())
            for name in sorted(lines))
    return compare_archive_contents(expected, actual)


def compare_archive_contents(
    expected: typing.Mapping[str, typing.Tuple[ArchiveMember, ...]],
    actual: typing.Mapping[str, typing.Tuple[ArchiveMember, ...]]
) -> typing.List[common.Finding]:
    """Compare expected target-owned object bytes with actual archive members."""
    findings = []
    for target_name in sorted(expected):
        wanted = {member.name: member.digest for member in expected[target_name]}
        observed = {member.name: member.digest for member in actual.get(target_name, ())}
        for member in sorted(wanted.keys() - observed.keys()):
            findings.append(common.Finding("archive", target_name, "missing_member", expected=f"{member} x1"))
        for member in sorted(observed.keys() - wanted.keys()):
            findings.append(common.Finding("archive", target_name, "unexpected_member", actual=f"{member} x1"))
        for member in sorted(wanted.keys() & observed.keys()):
            if wanted[member] != observed[member]:
                findings.append(
                    common.Finding("archive", target_name, "content_mismatch", expected=wanted[member],
                                   actual=observed[member], origin=member))
    return findings


def check_dynamic_dependencies(context: common.BoundaryContext) -> typing.List[common.Finding]:
    """Return findings for DT_NEEDED names outside the exact platform allowset."""
    readelf = context.cmake_snapshot.tool_paths["CMAKE_READELF"]
    targets = {target.name: target for target in context.cmake_snapshot.targets}
    actual = {}
    for policy in context.policy.targets:
        if policy.artifact_kind not in ("shared_library", "executable"):
            continue
        target = targets.get(policy.name)
        if target is None:
            if policy.required:
                raise common.BoundaryError(f"required artifact target is missing: {policy.name}")
            continue
        if len(target.artifacts) != 1 or not target.artifacts[0].is_file():
            raise common.BoundaryError(f"final artifact is missing: {policy.name}")
        libraries = []
        for line in common.run_lines(readelf, ["-d", target.artifacts[0]], 4194304):
            if "(NEEDED)" not in line:
                continue
            match = NEEDED_PATTERN.fullmatch(line.strip())
            if match is None:
                raise common.BoundaryError(f"malformed DT_NEEDED line: {line}")
            libraries.append(match.group("name"))
        actual[target.name] = libraries
    findings = []
    for target_name in sorted(actual):
        for library in sorted(actual[target_name]):
            if library not in context.policy.dynamic_names:
                findings.append(common.Finding("dynamic", target_name, "unadmitted_needed", actual=library))
    return findings


def check_undefined_symbols(context: common.BoundaryContext) -> typing.List[common.Finding]:
    """Return findings for unresolved symbols that match prohibited policy rules."""
    nm = context.cmake_snapshot.tool_paths["CMAKE_NM"]
    targets = {target.name: target for target in context.cmake_snapshot.targets}
    actual = {}
    for policy in context.policy.targets:
        target = targets.get(policy.name)
        if target is None:
            if policy.required:
                raise common.BoundaryError(f"required artifact target is missing: {policy.name}")
            continue
        if len(target.artifacts) != 1 or not target.artifacts[0].is_file():
            raise common.BoundaryError(f"root artifact is missing: {policy.name}")
        arguments = (["-u", target.artifacts[0]] if policy.artifact_kind == "static_archive"
                     else ["-D", "--undefined-only", target.artifacts[0]])
        symbols = []
        for line in common.run_lines(nm, arguments, 4194304):
            stripped = line.strip()
            if not stripped or stripped.endswith(":"):
                continue
            fields = stripped.split()
            if len(fields) < 2 or fields[-2] not in ("U", "w", "v") or not _valid_symbol_prefix(fields[:-2]):
                raise common.BoundaryError(f"malformed undefined-symbol line: {stripped}")
            symbol = fields[-1].split("@", 1)[0]
            if not symbol:
                raise common.BoundaryError("empty undefined symbol")
            symbols.append(symbol)
        actual[target.name] = symbols
    return compare_undefined_symbols(context.policy.symbol_rules, actual)


def compare_undefined_symbols(expected: typing.Sequence[common.Rule],
                              actual: typing.Mapping[str, typing.Sequence[str]]) -> typing.List[common.Finding]:
    """Compare unresolved symbols with target-scoped prohibited rules."""
    findings = []
    for target_name in sorted(actual):
        for symbol in sorted(actual[target_name]):
            for rule in expected:
                if rule.targets and target_name not in rule.targets:
                    continue
                if rule.expression.fullmatch(symbol) is not None:
                    findings.append(common.Finding("symbols", target_name, rule.id, actual=symbol))
    return findings


def expand_response(tokens: typing.Sequence[str], working_directory: pathlib.Path,
                    build_root: pathlib.Path) -> typing.List[str]:
    """Args: tokens/roots. Returns: expanded tokens. Raises: BoundaryError for unsafe response evidence."""
    expanded, seen = [], False
    for token in tokens:
        if not token.startswith("@"):
            expanded.append(token)
            continue
        if seen or len(token) == 1:
            raise common.BoundaryError("nested or multiple link response files are unsupported")
        text = common.read_text((working_directory / token[1:]).resolve(strict=False), build_root,
                                                1048576)
        try:
            response = shlex.split(text, posix=True)
        except ValueError as error:
            raise common.BoundaryError(
                f"malformed link response quoting: {error}") from error
        if any(value.startswith("@") or value.startswith("-Wl,@") for value in response):
            raise common.BoundaryError("nested link response files are unsupported")
        expanded.extend(response)
        seen = True
    return expanded


def classify(tokens: typing.Sequence[str], working_directory: pathlib.Path,
             build_root: pathlib.Path) -> LinkObservation:
    """Args: tokens/roots. Returns: observation. Raises: BoundaryError for unknown command forms."""
    output, input_tokens = extract_output(tokens, working_directory)
    inputs = []
    for token in input_tokens[1:]:
        inputs.extend(classify_token(token, working_directory, build_root))
    return LinkObservation(output, tuple(inputs))
def extract_output(tokens: typing.Sequence[str], working_directory: pathlib.Path) \
        -> typing.Tuple[pathlib.Path, typing.List[str]]:
    """Args: tokens/working directory. Returns: output/rest. Raises: BoundaryError unless one output exists."""
    indexes = [index for index, token in enumerate(tokens) if token == "-o"]
    if len(indexes) != 1 or indexes[0] + 1 >= len(tokens):
        raise common.BoundaryError(
            "link output must occur exactly once with a value")
    index = indexes[0]
    output = _resolve(tokens[index + 1], working_directory)
    return output, list(tokens[:index]) + list(tokens[index + 2:])


def classify_token(token: str, working_directory: pathlib.Path, build_root: pathlib.Path) -> typing.List[LinkInput]:
    """Args: token/roots. Returns: inputs. Raises: BoundaryError for shell or unknown input forms."""
    if token in SHELL_TOKENS or "$(" in token or "`" in token:
        raise common.BoundaryError("shell operators are unsupported in link.txt")
    if is_compiler_control(token):
        return []
    if token == "-Xlinker" or token.startswith(("-Wl,@", "-V", "-lang")):
        raise common.BoundaryError(f"unsupported link control form: {token}")
    if token.startswith("-Wl,"):
        return classify_linker_option(token, working_directory, build_root)
    if token.startswith("-l") or token.startswith("-L") or token == "-pthread" or token.startswith("--"):
        return [LinkInput("external_flag", token)]
    if token.endswith((".o", ".a")) or ".so" in token:
        return [classify_path(token, working_directory, build_root)]
    raise common.BoundaryError(f"unsupported link input token: {token}")


def is_compiler_control(token: str) -> bool:
    """Args: token. Returns: whether it is a supported non-input compiler control."""
    patterns = (r"-O(?:[0-3]|g|s)", r"-[DU][A-Za-z_][A-Za-z0-9_]*(?:=[A-Za-z0-9_.+-]+)?",
                r"-m(?:arch|cpu|tune|abi)=[A-Za-z0-9_.+-]+",
                r"--sysroot=(?:/[A-Za-z0-9_.+-]+)+", r"-Wc,-isysroot,(?:/[A-Za-z0-9_.+-]+)+",
                r"-V[0-9]+\.[0-9]+\.[0-9]+,gcc_ntoaarch64le")
    return token in CONTROL_FLAGS or any(re.fullmatch(pattern, token) is not None for pattern in patterns)


def classify_linker_option(token: str, working_directory: pathlib.Path,
                           build_root: pathlib.Path) -> typing.List[LinkInput]:
    """Args: linker token/roots. Returns: inputs. Raises: BoundaryError for unsafe payloads."""
    payloads = token[4:].split(",")
    if any(not value for value in payloads) or any(value.startswith("@") for value in payloads):
        raise common.BoundaryError(f"unsupported linker payload: {token}")
    if len(payloads) == 2 and payloads[0] == "--dependency-file":
        _require_build_path(payloads[1], working_directory, build_root, payloads[0])
        return []
    if len(payloads) == 1 and payloads[0].startswith("--dependency-file="):
        _require_build_path(payloads[0].split("=", 1)[1], working_directory, build_root, "--dependency-file")
        return []
    if len(payloads) == 2 and payloads[0] in ("-rpath", "--rpath"):
        paths = payloads[1].split(":")
        if any(not path for path in paths):
            raise common.BoundaryError("empty runtime search path")
        return [LinkInput("rpath", _resolve(path, working_directory)) for path in paths]
    if len(payloads) == 2 and payloads[0] in ("-soname", "--soname", "-rpath-link", "--rpath-link"):
        return [LinkInput("external_flag", f"{payloads[0]}={payloads[1]}")]
    inputs = []
    for payload in payloads:
        path_like = payload.endswith((".o", ".a")) or ".so" in payload
        inputs.append(classify_path(payload, working_directory, build_root)
                      if path_like else LinkInput("external_flag", payload))
    return inputs


def classify_path(value: str, working_directory: pathlib.Path, build_root: pathlib.Path) -> LinkInput:
    """Args: value/roots. Returns: classified canonical path input."""
    resolved = _resolve(value, working_directory)
    kind = "project_path" if common.is_below(resolved, build_root) else "external_path"
    return LinkInput(kind, resolved)


def load_link_command(build_root: pathlib.Path, target_name: str,
                      target_build_directory: pathlib.Path) -> LinkObservation:
    """Args: build/target paths. Returns: observation. Raises: BoundaryError for invalid link.txt."""
    path = target_build_directory / "CMakeFiles" / (target_name + ".dir") / "link.txt"
    text = common.read_text(path, build_root, 2097152)
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError as error:
        raise common.BoundaryError(f"malformed link.txt quoting: {error}") from error
    if not tokens:
        raise common.BoundaryError("link.txt is empty")
    return classify(expand_response(tokens, target_build_directory, build_root), target_build_directory, build_root)

def _dependency_closure(root: build_metadata.CMakeTarget,
                        targets_by_id: typing.Mapping[str, build_metadata.CMakeTarget]) \
        -> typing.List[build_metadata.CMakeTarget]:
    """Return transitive dependencies without the root output."""
    pending, selected = list(root.dependency_ids), set()
    while pending:
        target = targets_by_id[pending.pop(0)]
        if target.id not in selected:
            selected.add(target.id)
            pending.extend(target.dependency_ids)
    return [targets_by_id[target_id] for target_id in sorted(selected)]


def _require_build_path(value: str, working_directory: pathlib.Path,
                        build_root: pathlib.Path, option: str) -> None:
    """Require one non-input linker-control path to stay in the build tree."""
    if not common.is_below(_resolve(value, working_directory), build_root):
        raise common.BoundaryError(f"{option} path escapes build root")


def _resolve(value: str, working_directory: pathlib.Path) -> pathlib.Path:
    """Resolve a possibly relative command path without requiring existence."""
    path = pathlib.Path(value)
    return path.resolve(strict=False) if path.is_absolute() else (working_directory / path).resolve(strict=False)


def _finding(target: str, kind: str, expected: typing.Any,
             actual: typing.Any) -> common.Finding:
    """Create a link finding with normalized string values."""
    return common.Finding("link", target, kind, str(expected), str(actual))


def _is_admitted(value: str, expectation: LinkExpectation) -> bool:
    """Return whether an external path is admitted and not explicitly forbidden."""
    forbidden = any(rule.expression.fullmatch(value) is not None
                    for rule in expectation.forbidden_external_patterns)
    admitted = any(rule.expression.fullmatch(value) is not None for rule in expectation.external_patterns)
    return admitted and not forbidden


def _is_shared_library(path: pathlib.Path) -> bool:
    """Return whether a path name denotes an ELF shared library."""
    return path.name.endswith(".so") or ".so." in path.name


def _valid_symbol_prefix(fields: typing.Sequence[str]) -> bool:
    """Return whether optional GNU/LLVM nm prefix fields are recognized."""
    if len(fields) > 2:
        return False
    if len(fields) == 2:
        return fields[0].endswith(":") and re.fullmatch(r"[0-9A-Fa-f]+", fields[1]) is not None
    if len(fields) == 1:
        return fields[0].endswith(":") or re.fullmatch(r"[0-9A-Fa-f]+", fields[0]) is not None
    return True
