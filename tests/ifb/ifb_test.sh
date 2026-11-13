#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

# ifb_test.sh - trace-driven IFB correctness tiers.
#
# USAGE-BEGIN
#   quick: ifb_smoke.json -> reference + run + check
#   full : ifb_smoke, ifb_priority, ifb_context_cache, and ifb_multimodal on a VLM engine
#          -> reference + run + check
#
# Usage:
#   tests/ifb/ifb_test.sh --tier quick|full --engine <dir> [--engine <dir> ...]
#       [--mode mixed|prefillFirst|auto|singleCohort] [--out <dir>] [--python <exe>]
#       [--nanobind-dir <dir>] [--traces a,b] [--timeout <s>]
#
# An engine directory holding llm/ and visual/ is a VLM engine: llm/ is the engine and the directory
# itself is passed as --multimodal-engine-dir; ifb_multimodal runs only on such engines.
# --nanobind-dir defaults to $EDGELLM_NANOBIND_DIR, then the directory of $EDGELLM_PLUGIN_LIB, then
# <repo>/build/lib. LD_LIBRARY_PATH, EDGELLM_PLUGIN_LIB and EDGELLM_PLUGIN_PATH must point at the same build.
# The reference runs use ifb_scheduling_mode=singleCohort on the build under test.
# USAGE-END

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
TIER=quick
ENGINES=()
MODE=""
OUT=./ifb-test-out
PY=${PYTHON:-python3}
NANOBIND_DIR=${EDGELLM_NANOBIND_DIR:-}
TRACES=""
TIMEOUT=900

usage() { sed -n '/^# USAGE-BEGIN$/,/^# USAGE-END$/{/^# USAGE-/d;p;}' "$0"; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --tier) TIER=$2; shift 2 ;;
    --engine) ENGINES+=("$2"); shift 2 ;;
    --mode) MODE=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    --nanobind-dir) NANOBIND_DIR=$2; shift 2 ;;
    --traces) TRACES=$2; shift 2 ;;
    --timeout) TIMEOUT=$2; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

[ ${#ENGINES[@]} -gt 0 ] || { echo "at least one --engine is required" >&2; usage; }
case "$TIER" in quick|full) ;; *) echo "--tier must be quick or full" >&2; exit 2 ;; esac
case "$MODE" in ""|mixed|prefillFirst|auto|singleCohort) ;; *) echo "--mode must be mixed, prefillFirst, auto or singleCohort" >&2; exit 2 ;; esac

if [ -z "$NANOBIND_DIR" ]; then
  if [ -n "${EDGELLM_PLUGIN_LIB:-}" ]; then
    NANOBIND_DIR=$(dirname "$EDGELLM_PLUGIN_LIB")
  else
    NANOBIND_DIR=$REPO/build/lib
  fi
fi

if [ -n "$TRACES" ]; then
  IFS=, read -r -a TRACE_LIST <<< "$TRACES"
elif [ "$TIER" = quick ]; then
  TRACE_LIST=(ifb_smoke)
else
  TRACE_LIST=(ifb_smoke ifb_priority ifb_context_cache ifb_multimodal)
fi

mkdir -p "$OUT"
DRIVER=("$PY" "$SCRIPT_DIR/run_ifb_trace.py" --nanobind-dir "$NANOBIND_DIR" --timeout "$TIMEOUT")
CHECKER=("$PY" "$SCRIPT_DIR/check_ifb_correctness.py")
SUMMARY=()
FAILED=0

run_logged() {
  local log=$1; shift
  echo "+ $*" | tee -a "$log"
  "$@" 2>&1 | tee -a "$log"
  return "${PIPESTATUS[0]}"
}

for engine in "${ENGINES[@]}"; do
  ename=$(basename "$engine")
  mm_args=()
  has_visual=0
  if [ -d "$engine/llm" ] && [ -d "$engine/visual" ]; then
    has_visual=1
    mm_args=(--multimodal-engine-dir "$engine")
    engine=$engine/llm
  fi
  for trace in "${TRACE_LIST[@]}"; do
    tfile=$REPO/tests/test_cases/$trace.json
    if [ ! -f "$tfile" ]; then
      echo "skip $trace: $tfile not present"
      SUMMARY+=("$ename $trace SKIPPED(no trace)")
      continue
    fi
    if [ "$trace" = ifb_multimodal ] && [ $has_visual -eq 0 ]; then
      echo "skip $trace on $ename: no visual/ engine"
      SUMMARY+=("$ename $trace SKIPPED(no visual engine)")
      continue
    fi
    dir=$OUT/$ename
    mkdir -p "$dir"
    log=$dir/$trace.log
    : > "$log"
    tag=${MODE:-default}

    run_args=()
    [ -n "$MODE" ] && run_args+=(--scheduling-mode "$MODE")
    case "$MODE" in
      mixed|prefillFirst|singleCohort) run_args+=(--expect-policy "$MODE") ;;
    esac
    # pause preemption requires mixed; under prefillFirst and singleCohort only the other variants apply.
    if [ "$trace" = ifb_priority ] && { [ "$MODE" = prefillFirst ] || [ "$MODE" = singleCohort ]; }; then
      run_args+=(--variant free_slot_other_length,reserved_other_length)
    fi

    # A reference that did not start or left requests incomplete fails the trace, whatever the checker compared.
    reference_ok=1
    run_logged "$log" "${DRIVER[@]}" --trace "$tfile" --engine-dir "$engine" "${mm_args[@]}" --mode reference \
      --out "$dir/$trace.reference.json" || { reference_ok=0; echo "reference run failed" | tee -a "$log"; }
    # The checker reports the run's incomplete requests beside its other criteria.
    run_logged "$log" "${DRIVER[@]}" --trace "$tfile" --engine-dir "$engine" "${mm_args[@]}" --mode run "${run_args[@]}" \
      --out "$dir/$trace.run.$tag.json" || echo "run reported incomplete requests" | tee -a "$log"
    if run_logged "$log" "${CHECKER[@]}" --run "$dir/$trace.run.$tag.json" \
        --reference "$dir/$trace.reference.json" --json-out "$dir/$trace.check.$tag.json" && [ $reference_ok -eq 1 ]; then
      SUMMARY+=("$ename $trace mode=$tag PASS")
    else
      SUMMARY+=("$ename $trace mode=$tag FAIL (see $log)")
      FAILED=1
    fi
  done
done

echo "==== ifb_test.sh tier=$TIER mode=${MODE:-default} out=$OUT"
printf '%s\n' "${SUMMARY[@]}"
exit $FAILED
