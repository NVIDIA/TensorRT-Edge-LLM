#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# run_coverage.sh - Build with gcov instrumentation, run unit tests, and
# generate coverage reports suitable for SonarQube analysis.
#
# Usage:
#   ./scripts/run_coverage.sh [--trt-package-dir <path>] [--cuda-version <ver>]
#                              [--build-dir <dir>] [--gtest-filter <filter>]
#
# Environment variables (alternative to flags):
#   TRT_PACKAGE_DIR   Path to TensorRT package (required)
#   CUDA_VERSION      CUDA version (default: 12.8)
#   ENABLE_CUTE_DSL   CuTe DSL selection (default: fmha)
#   CUTE_DSL_ARTIFACT_TAG  Required artifact tag when selection is ambiguous
#
# After a successful run the build directory will contain:
#   - sonarqube-coverage.xml  (SonarQube generic coverage format)
#   - coverage.xml            (Cobertura XML)
#   - coverage.html           (HTML report for local viewing)
#
# SonarQube picks up coverage from:
#   sonar.coverageReportPaths=<build-dir>/sonarqube-coverage.xml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BUILD_DIR="${PROJECT_ROOT}/build_coverage"
TRT_PACKAGE_DIR="${TRT_PACKAGE_DIR:-}"
CUDA_VERSION="${CUDA_VERSION:-12.8}"
ENABLE_CUTE_DSL="${ENABLE_CUTE_DSL:-fmha}"
CUTE_DSL_ARTIFACT_TAG="${CUTE_DSL_ARTIFACT_TAG:-}"
GTEST_FILTER="${GTEST_FILTER:-*}"
JOBS="$(nproc 2>/dev/null || echo 8)"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --trt-package-dir)
            TRT_PACKAGE_DIR="$2"; shift 2 ;;
        --cuda-version)
            CUDA_VERSION="$2"; shift 2 ;;
        --build-dir)
            BUILD_DIR="$2"; shift 2 ;;
        --gtest-filter)
            GTEST_FILTER="$2"; shift 2 ;;
        --jobs|-j)
            JOBS="$2"; shift 2 ;;
        -h|--help)
            head -28 "$0" | tail -20; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${TRT_PACKAGE_DIR}" ]]; then
    echo "ERROR: TRT_PACKAGE_DIR is not set." >&2
    echo "  Pass --trt-package-dir <path> or export TRT_PACKAGE_DIR." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 1 — Configure (CMake)
# ---------------------------------------------------------------------------
echo "==> Configuring coverage build in ${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DBUILD_UNIT_TESTS=ON \
    -DENABLE_COVERAGE=ON \
    -DENABLE_CUTE_DSL="${ENABLE_CUTE_DSL}" \
    -DCUTE_DSL_ARTIFACT_TAG="${CUTE_DSL_ARTIFACT_TAG}" \
    -DTRT_PACKAGE_DIR="${TRT_PACKAGE_DIR}" \
    -DCUDA_CTK_VERSION="${CUDA_VERSION}"

# ---------------------------------------------------------------------------
# Step 2 — Build
# ---------------------------------------------------------------------------
echo "==> Building with coverage instrumentation (${JOBS} jobs)"
cmake --build "${BUILD_DIR}" --target unitTests -j "${JOBS}"

# ---------------------------------------------------------------------------
# Step 3 — Clear stale coverage data
# ---------------------------------------------------------------------------
echo "==> Clearing old gcov data"
find "${BUILD_DIR}" -name '*.gcda' -delete 2>/dev/null || true
# Remove any .gcno/.gcda files left over from CUDA compilations. nvcc
# generates gcno files that reference ephemeral /tmp/ stub sources which
# cannot be resolved by gcov, so these must be excluded.
find "${BUILD_DIR}" -name '*.cu.gcno' -delete 2>/dev/null || true
find "${BUILD_DIR}" -name '*.cu.gcda' -delete 2>/dev/null || true

# ---------------------------------------------------------------------------
# Step 4 — Run unit tests
# ---------------------------------------------------------------------------
echo "==> Running unit tests (filter: ${GTEST_FILTER})"
# Every group contributes to the same gcda set, so coverage is the union of all
# of them. That is also why this runs serially: the groups share edgellmCore's
# objects, so concurrent processes would be merging the same .gcda files.
# GTEST_FILTER and GTEST_OUTPUT reach the executables through the environment
# because ctest drives them, not this script.
#
# The reports come from gtest rather than from `ctest --output-junit`, for two
# reasons. --output-junit needs CMake 3.21 and this project declares 3.20
# (--test-dir is fine at 3.20). And ctest reports one entry per registered test,
# which here is one per group: the per-test-case detail that the JUnit consumers
# read would be replaced by ten rows. A trailing slash makes gtest write one
# file per executable, named after it.
TEST_REPORT_DIR="${BUILD_DIR}/test_results"
rm -rf "${TEST_REPORT_DIR}"
mkdir -p "${TEST_REPORT_DIR}"
GTEST_FILTER="${GTEST_FILTER}" \
    GTEST_OUTPUT="xml:${TEST_REPORT_DIR}/" \
    ctest --test-dir "${BUILD_DIR}" \
          --output-on-failure \
    || TEST_EXIT=$?

if [[ "${TEST_EXIT:-0}" -ne 0 ]]; then
    echo "WARNING: Some tests failed (exit code ${TEST_EXIT}). Coverage data is still valid."
fi

# ---------------------------------------------------------------------------
# Step 5 — Generate coverage reports
# ---------------------------------------------------------------------------
echo "==> Generating coverage reports"

# gcovr is required to produce the SonarQube generic coverage XML.
# --gcov-ignore-errors guards against any residual CUDA .gcno files that
# reference missing /tmp/ stub sources.
if ! command -v gcovr &>/dev/null; then
    echo "ERROR: gcovr not found — required to generate SonarQube coverage report." >&2
    echo "       Install with: pip install gcovr" >&2
    exit 1
fi

GCOVR_COMMON=(
    --root "${PROJECT_ROOT}"
    --filter "${PROJECT_ROOT}/cpp/"
    --filter "${PROJECT_ROOT}/unittests/"
    --exclude "${PROJECT_ROOT}/3rdParty/"
    --exclude '.*\.cu$'
    --gcov-executable gcov
    --gcov-ignore-errors=no_working_dir_found
    --gcov-ignore-parse-errors=all
)

echo "    Generating SonarQube generic coverage XML"
gcovr "${GCOVR_COMMON[@]}" \
    --sonarqube "${BUILD_DIR}/sonarqube-coverage.xml"

echo "    Generating Cobertura XML report"
gcovr "${GCOVR_COMMON[@]}" \
    --xml-pretty \
    --output "${BUILD_DIR}/coverage.xml"

echo "    Generating HTML report"
gcovr "${GCOVR_COMMON[@]}" \
    --html-details "${BUILD_DIR}/coverage.html"

echo "    SonarQube report : ${BUILD_DIR}/sonarqube-coverage.xml"
echo "    Cobertura XML    : ${BUILD_DIR}/coverage.xml"
echo "    HTML report      : ${BUILD_DIR}/coverage.html"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
GCNO_COUNT=$(find "${BUILD_DIR}" -name '*.gcno' | wc -l)
GCDA_COUNT=$(find "${BUILD_DIR}" -name '*.gcda' | wc -l)

echo ""
echo "=== Coverage Summary ==="
echo "  Build directory    : ${BUILD_DIR}"
echo "  .gcno files        : ${GCNO_COUNT}"
echo "  .gcda files        : ${GCDA_COUNT}"
echo "  Test results       : ${BUILD_DIR}/test_results/ (one XML per group)"
echo "  SonarQube coverage : ${BUILD_DIR}/sonarqube-coverage.xml"
echo ""
echo "To analyze with SonarQube, ensure sonar-project.properties contains:"
echo "  sonar.coverageReportPaths=${BUILD_DIR}/sonarqube-coverage.xml"
echo "  sonar.cfamily.compile-commands=${BUILD_DIR}/compile_commands.json"
echo ""
echo "Then run: sonar-scanner"
