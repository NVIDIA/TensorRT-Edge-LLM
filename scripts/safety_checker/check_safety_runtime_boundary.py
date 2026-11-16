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
"""Validate configured and built Safety Runtime boundaries."""

import argparse
import pathlib
import sys
import typing

import artifact_checks
import build_metadata
import common
import source_checks

CheckFunction = typing.Callable[[common.BoundaryContext],
                                typing.List[common.Finding]]


def run(arguments: typing.Sequence[str]) -> int:
    """Parse, load, dispatch, report, and return one phase status.

    Args:
        arguments: Command-line arguments excluding the executable name.

    Returns:
        Zero when all comparisons pass, otherwise one.
    """
    options = parse_arguments(arguments)
    try:
        context = load_context(options)
    except common.BoundaryError as error:
        return report_context_error(options, error)
    findings = run_phase(context)
    return write_result(options.report_path, options.phase, findings)


def report_context_error(options: common.Options,
                         error: common.BoundaryError) -> int:
    """Args: options/error. Returns: failure after publishing one context finding."""
    finding = common.Finding("context", kind="evidence", actual=str(error))
    return write_result(options.report_path, options.phase, [finding])


def run_phase(context: common.BoundaryContext) -> typing.List[common.Finding]:
    """Args: loaded context. Returns: findings from exactly one selected phase."""
    if context.options.phase == "configured":
        return run_configured_checks(context)
    return run_artifact_checks(context)


def run_configured_checks(
        context: common.BoundaryContext) -> typing.List[common.Finding]:
    """Args: context. Returns: target/source findings; publishes a plan only when empty."""
    findings = exercise_check("target_graph", source_checks.check_target_graph,
                              context)
    findings.extend(
        exercise_check("source", source_checks.check_sources, context))
    if not findings:
        build_metadata.write_configured_plan(context)
    return findings


def run_artifact_checks(
        context: common.BoundaryContext) -> typing.List[common.Finding]:
    """Args: context. Returns: drift only, or all realized artifact findings after drift passes."""
    findings = exercise_check("plan_drift", build_metadata.check_plan_drift,
                              context)
    if findings:
        return findings
    checks = (("source", source_checks.check_sources),
              ("archive", artifact_checks.check_archive_contents),
              ("link", artifact_checks.check_link_inputs),
              ("dynamic", artifact_checks.check_dynamic_dependencies),
              ("symbols", artifact_checks.check_undefined_symbols))
    for check_name, check in checks:
        findings.extend(exercise_check(check_name, check, context))
    return findings


def exercise_check(
        check_name: str, check: CheckFunction,
        context: common.BoundaryContext) -> typing.List[common.Finding]:
    """Run one check and translate malformed evidence into a finding."""
    try:
        return check(context)
    except common.BoundaryError as error:
        return [common.Finding(check_name, kind="evidence", actual=str(error))]


def write_result(report_path: pathlib.Path, phase: str,
                 findings: typing.Sequence[common.Finding]) -> int:
    """Args: report path/phase/findings. Returns: zero only after publishing an empty report."""
    common.write_report(report_path, phase, findings)
    return 0 if not findings else 1


def parse_arguments(arguments: typing.Sequence[str]) -> common.Options:
    """Args: CLI values. Returns: canonical options. Raises: BoundaryError for unsupported or absent roots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--phase", required=True)
    parsed = parser.parse_args(arguments)
    if parsed.platform not in common.SUPPORTED_PLATFORMS:
        raise common.BoundaryError("platform must be x86, d7l, or d7qs")
    if parsed.phase not in ("configured", "artifacts"):
        raise common.BoundaryError("phase must be configured or artifacts")
    repo_root = pathlib.Path(parsed.repo_root).resolve(strict=False)
    build_root = pathlib.Path(parsed.build_dir).resolve(strict=False)
    if not repo_root.is_dir() or not build_root.is_dir():
        raise common.BoundaryError("repo-root and build-dir must exist")
    boundary_root = build_root / "safety-boundary"
    return common.Options(repo_root, build_root, parsed.platform,
                          parsed.configuration, parsed.phase,
                          boundary_root / "plan.json",
                          boundary_root / "report.json")


def load_context(options: common.Options) -> common.BoundaryContext:
    """Args: options. Returns: context. Raises: BoundaryError for invalid policy, CMake, or plan evidence."""
    policy = common.load_policy(options.repo_root, options.platform)
    compile_target_names = frozenset(name for target in policy.targets
                                     for name in target.admitted_targets)
    snapshot = build_metadata.load_snapshot(options.repo_root,
                                            options.build_root,
                                            options.configuration,
                                            compile_target_names)
    plan = build_metadata.load_configured_plan(options)
    return common.BoundaryContext(options, policy, snapshot, plan)


def main(arguments: typing.Optional[typing.Sequence[str]] = None) -> int:
    """Args: optional CLI values. Returns: boundary-checker process status."""
    return run(sys.argv[1:] if arguments is None else arguments)


if __name__ == "__main__":
    sys.exit(main())
