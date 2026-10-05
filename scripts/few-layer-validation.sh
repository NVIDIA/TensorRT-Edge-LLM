#!/usr/bin/env bash
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
#
# End-to-end few-layer numeric validation.
#
# Runs the whole smoke test for one model:
#   1. PyTorch golden            -> golden.safetensors
#   2. EdgeLLM export (N layers) -> export/llm/model.onnx
#   3. EdgeLLM build             -> engine/llm.engine
#   4. EdgeLLM inference (+dump)  -> edgellm_dump/edgellm_dump.safetensors
#   5. compare golden vs edgellm -> PASS/FAIL (this script's exit code)
#
# Everything is parameterized so it is portable (no per-user absolute paths) and
# CI-friendly. Paths are resolved relative to the repo, intermediates go in a temp
# workdir, and max-generate-length / batch-size are read from the input JSON so the
# golden and the runtime stay aligned.
#
# Usage:
#   scripts/few-layer-validation.sh --model PATH [options]
#
# Options (--model is required; the rest are optional):
#   --model PATH        Checkpoint dir or HF repo id to validate (REQUIRED).
#   --num-layers N      Number of leading decoder layers to validate (default: 4).
#   --input-file JSON   Requests JSON, llm_inference format
#                       (default: <repo>/tests/test_cases/ragged_batch.json).
#   --build-dir DIR     CMake build dir with the llm_build/llm_inference binaries and
#                       the plugin .so (default: <repo>/build).
#   --python BIN        Python used for the golden + ONNX export (default: python3,
#                       i.e. the active venv; needs torch + transformers + the TRT wheel).
#   --workdir DIR       Where to put intermediates (default: a fresh mktemp dir).
#   --keep              Do not delete the workdir on exit (for debugging).
#   --cos X             Min cosine similarity threshold (default: 0.99).
#   --scale-tol X       Max |norm(edgellm)/norm(golden) - 1| per tensor (default: 0.25).
#                       Cosine is scale-invariant, so this is what catches a dropped
#                       scale factor.
#   --atol X            allclose atol (default: 2e-2, reported only).
#   --rtol X            allclose rtol (default: 2e-2, reported only).
#   --max-input-len N   Engine maxInputLen. Default: derived from the prompts, rounded up
#                       to a KV page and floored at 128. Engine build parameters can
#                       affect accuracy, so it stays overridable.
#   --max-kv-cache-capacity N
#                       Engine maxKVCacheCapacity. Default: derived the same way from
#                       prompt + generate length, floored at 256. Drives the KV-cache
#                       sequence dim, so the dump size scales with it (see Dump size).
#   --target-sm N       Target GPU compute capability used for target-specific NVFP4
#                       dense and MoE layouts (for example, 100, 110, or 120).
#   --mtp               Also export/build the checkpoint's MTP draft and run the engine with
#                       speculative decoding. The PyTorch golden stays vanilla on purpose: MTP is
#                       supposed to be output-equivalent, so the base model's committed KV and
#                       logits must match vanilla token for token. Requires an MTP-capable
#                       checkpoint (Qwen3.5 and friends).
#   --spec-draft-step N Draft tokens proposed per round with --mtp (default: 3).
#   --context-reuse     Run the engine with the managed context cache enabled. Point --input-file
#                       at requests that share a long prefix (tests/test_cases/llm_context_reuse.json)
#                       and set batch_size to 1 so each one is its own request: the first populates
#                       the cache and the rest reuse it. The golden prefills every request in full,
#                       which is the invariant -- a reused prefix must equal a freshly computed one.
#   --no-quantize-activations
#                       Export both sides with the activation Q-DQ dropped (W4A16 / W8A16 instead
#                       of the checkpoint's W4A4 / W8A8). Diagnostic: it isolates how much of a
#                       quantized model's error comes from the activation rather than the weight.
#                       Passed to the golden and the export together -- the two must agree, or the
#                       comparison is measuring the flag instead of the engine.
#   --verbose           Print per-tensor cos / max_abs for every round (passed to the comparison).
#
# Every stage reports its wall clock, followed by a summary table before the RESULT line.
#   -h | --help         Show this help.
set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root (this script lives in <repo>/scripts/).
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Make the repo importable for the golden/export Python without requiring an
# editable install (`pip install -e .`). CI runs the golden as a script, so its
# sys.path[0] is tests/, not the repo root -- without this the quantized golden's
# `import tensorrt_edgellm` would fail. Harmless when the package is installed.
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL=""
NUM_LAYERS=4
INPUT_FILE="${REPO}/tests/test_cases/ragged_batch.json"
BUILD_DIR="${REPO}/build"
PYBIN="python3"
WORKDIR=""
KEEP=0
COS=0.99
SCALE_TOL=0.25
ATOL=2e-2
RTOL=2e-2
# 0 until the caller overrides them; otherwise they are derived from the prompts once the
# golden has tokenized them (see "engine bounds" below).
MAX_INPUT_LEN=0
MAX_KV_CACHE_CAPACITY=0
MAX_KV_CACHE_CAPACITY_OVERRIDDEN=0
TARGET_SM=""
VERBOSE=0
QUANTIZE_ACTIVATIONS=1
MTP=0
SPEC_DRAFT_STEP=3
CONTEXT_REUSE=0
# Hybrid (Mamba / Gated DeltaNet) reuse refuses to start without a recurrent snapshot slot, and a
# partial-KV pool is what lets a hit land off a page boundary. 256 MB each is ample for a few-layer
# truncation; the runtime sizes its record count itself.
CONTEXT_CACHE_POOL_BYTES=268435456

# Print the leading doc block (from the title down to the first non-comment line).
usage() { awk '/^# End-to-end/{p=1} p&&!/^#/{exit} p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL="$2"; shift 2 ;;
    --num-layers)  NUM_LAYERS="$2"; shift 2 ;;
    --input-file)  INPUT_FILE="$2"; shift 2 ;;
    --build-dir)   BUILD_DIR="$2"; shift 2 ;;
    --python)      PYBIN="$2"; shift 2 ;;
    --workdir)     WORKDIR="$2"; shift 2 ;;
    --keep)        KEEP=1; shift ;;
    --cos)         COS="$2"; shift 2 ;;
    --scale-tol)   SCALE_TOL="$2"; shift 2 ;;
    --atol)        ATOL="$2"; shift 2 ;;
    --rtol)        RTOL="$2"; shift 2 ;;
    --max-input-len)          MAX_INPUT_LEN="$2"; shift 2 ;;
    --max-kv-cache-capacity)  MAX_KV_CACHE_CAPACITY="$2"; MAX_KV_CACHE_CAPACITY_OVERRIDDEN=1; shift 2 ;;
    --target-sm)              TARGET_SM="$2"; shift 2 ;;
    --mtp)         MTP=1; shift ;;
    --context-reuse) CONTEXT_REUSE=1; shift ;;
    --spec-draft-step) SPEC_DRAFT_STEP="$2"; shift 2 ;;
    --no-quantize-activations) QUANTIZE_ACTIVATIONS=0; shift ;;
    --verbose)     VERBOSE=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

LLM_BUILD="${BUILD_DIR}/examples/llm/llm_build"
LLM_INFER="${BUILD_DIR}/examples/llm/llm_inference"
PLUGIN="${BUILD_DIR}/libNvInfer_edgellm_plugin.so"

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
fail() { echo "[few-layer] ERROR: $*" >&2; exit 2; }

[[ -n "${MODEL}" ]] || fail "--model is required: pass a checkpoint dir or HF repo id"
[[ -e "${MODEL}" || "${MODEL}" == */* ]] || fail "model '${MODEL}' not found (pass --model)"
[[ -f "${INPUT_FILE}" ]] || fail "input file not found: ${INPUT_FILE}"
[[ -x "${LLM_BUILD}" ]] || fail "llm_build not found at ${LLM_BUILD} (build first, or pass --build-dir)"
[[ -x "${LLM_INFER}" ]] || fail "llm_inference not found at ${LLM_INFER} (build first, or pass --build-dir)"
[[ -f "${PLUGIN}" ]] || fail "plugin not found at ${PLUGIN}"

# ---------------------------------------------------------------------------
# Stage timing
# ---------------------------------------------------------------------------
# Each stage's wall clock is printed as it finishes and summarized at the end.
# The breakdown is what tells you where a CI timeout went: export and build
# dominate and scale with the *checkpoint*, not with --num-layers (a 4-layer
# MoE still loads and repacks every expert weight in those layers).
STAGE_NAMES=()
STAGE_SECS=()
_stage_name=""
_stage_t0=0

stage_begin() {
  _stage_name="$1"
  _stage_t0="$(date +%s.%N)"
  echo "[few-layer] === ${_stage_name} ==="
}

stage_end() {
  local dt
  dt="$(awk -v a="$(date +%s.%N)" -v b="${_stage_t0}" 'BEGIN { printf "%.1f", a - b }')"
  STAGE_NAMES+=("${_stage_name}")
  STAGE_SECS+=("${dt}")
  echo "[few-layer] --- ${_stage_name}: ${dt}s"
}

print_timing() {
  local i total=0
  echo
  echo "[few-layer] stage wall clock:"
  for i in "${!STAGE_NAMES[@]}"; do
    printf '[few-layer]   %-26s %8.1fs\n' "${STAGE_NAMES[$i]}" "${STAGE_SECS[$i]}"
    total="$(awk -v a="${total}" -v b="${STAGE_SECS[$i]}" 'BEGIN { printf "%.1f", a + b }')"
  done
  printf '[few-layer]   %-26s %8.1fs\n' "total" "${total}"
}

# Read batch size and max generate length from the input JSON so both sides agree.
read -r BATCH_SIZE MAX_GEN < <("${PYBIN}" - "${INPUT_FILE}" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
print(int(c.get("batch_size", len(c["requests"]))), int(c.get("max_generate_length", 6)))
PY
)
[[ -n "${BATCH_SIZE}" && -n "${MAX_GEN}" ]] \
  || fail "could not parse batch_size / max_generate_length from ${INPUT_FILE}"
DECODE_ROUNDS=$(( MAX_GEN - 1 ))

# Speculative decoding (--mtp). A linear MTP chain verifies the pending token plus every drafted
# one, so the runtime requires verifySize == draftStep + 1; the engines are built with the same
# bound. The golden is unaffected -- it stays vanilla, which is the invariant being checked.
MTP_EXPORT_ARGS=()
# One flag, both sides: the golden and the export must build the same graph regime.
WEIGHT_ONLY_ARGS=()
if [[ "${QUANTIZE_ACTIVATIONS}" -eq 0 ]]; then
  WEIGHT_ONLY_ARGS=(--no-quantize-activations)
fi
MTP_BASE_BUILD_ARGS=()
MTP_INFER_ARGS=()
REUSE_INFER_ARGS=()
if [[ "${CONTEXT_REUSE}" -eq 1 ]]; then
  REUSE_INFER_ARGS=(--enableContextReuse
                    --contextCacheRecurrentSnapshotPoolBytes "${CONTEXT_CACHE_POOL_BYTES}"
                    --contextCachePartialKVSnapshotPoolBytes "${CONTEXT_CACHE_POOL_BYTES}")
fi
SPEC_VERIFY_SIZE=$(( SPEC_DRAFT_STEP + 1 ))
if [[ "${MTP}" -eq 1 ]]; then
  MTP_EXPORT_ARGS=(--mtp)
  MTP_BASE_BUILD_ARGS=(--maxVerifyTreeSize "${SPEC_VERIFY_SIZE}" --specBase)
  MTP_INFER_ARGS=(--specDecode --specDraftTopK 1 --specDraftStep "${SPEC_DRAFT_STEP}"
                  --specVerifySize "${SPEC_VERIFY_SIZE}")
fi

# Round @1 up to a multiple of @2.
round_up() { echo $(( ( ($1 + $2 - 1) / $2 ) * $2 )); }

# Workdir (temp by default; cleaned up unless --keep).
if [[ -z "${WORKDIR}" ]]; then WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/few-layer-XXXXXX")"; fi
mkdir -p "${WORKDIR}"
cleanup() { if [[ "${KEEP}" -eq 0 ]]; then rm -r "${WORKDIR}"; fi; }
trap cleanup EXIT

GOLDEN="${WORKDIR}/golden.safetensors"
EXPORT_DIR="${WORKDIR}/export"
ENGINE_DIR="${WORKDIR}/engine"
DUMP_DIR="${WORKDIR}/edgellm_dump"

echo "[few-layer] repo        : ${REPO}"
echo "[few-layer] model       : ${MODEL}"
echo "[few-layer] num-layers  : ${NUM_LAYERS}  (dump layers 0-$(( NUM_LAYERS - 1 )))"
echo "[few-layer] input       : ${INPUT_FILE}  (batch=${BATCH_SIZE}, max_gen=${MAX_GEN})"
if [[ "${MTP}" -eq 1 ]]; then
  echo "[few-layer] decoding    : MTP speculative (draft step ${SPEC_DRAFT_STEP}, verify ${SPEC_VERIFY_SIZE}) vs a vanilla golden"
fi
if [[ "${CONTEXT_REUSE}" -eq 1 ]]; then
  echo "[few-layer] cache       : context reuse ON vs a golden that prefills every request in full"
fi
echo "[few-layer] python      : ${PYBIN}"
echo "[few-layer] workdir     : ${WORKDIR}"
[[ -n "${TARGET_SM}" ]] && echo "[few-layer] target SM   : ${TARGET_SM}"
echo

# ---------------------------------------------------------------------------
# 1. PyTorch golden
# ---------------------------------------------------------------------------
stage_begin "1/5 PyTorch golden"
"${PYBIN}" "${REPO}/tests/golden_layer_dump.py" \
  --ckpt "${MODEL}" \
  --input-file "${INPUT_FILE}" \
  --num-layers "${NUM_LAYERS}" \
  --decode-rounds "${DECODE_ROUNDS}" \
  --out "${GOLDEN}" \
  ${WEIGHT_ONLY_ARGS[@]+"${WEIGHT_ONLY_ARGS[@]}"}

# Teacher forcing: extract the golden's per-sequence tokens so EdgeLLM replays the exact same
# sequence (decouples the numeric comparison from greedy argmax stability). One line per sequence.
# Engine bounds: unless overridden, size them to the prompts the golden just tokenized.
# Hard-coding them means a case breaks the day someone lengthens a shared requests JSON --
# which is exactly what happened to tests/test_cases/llm_context_reuse.json. Round up to the
# KV page size, and keep the historical floors so short-prompt runs build the same engine
# (and dump the same size) as before.
GOLDEN_SEQ_LEN="$("${PYBIN}" - "${GOLDEN}" <<'PY'
import sys
from safetensors import safe_open
with safe_open(sys.argv[1], framework="pt") as f:
    print(f.metadata()["seq_len_prefill"])
PY
)"
[[ -n "${GOLDEN_SEQ_LEN}" ]] || fail "could not read seq_len_prefill from ${GOLDEN}"
if [[ "${MAX_INPUT_LEN}" -eq 0 ]]; then
  MAX_INPUT_LEN="$(round_up "${GOLDEN_SEQ_LEN}" 128)"
  (( MAX_INPUT_LEN < 128 )) && MAX_INPUT_LEN=128
fi
if [[ "${MAX_KV_CACHE_CAPACITY}" -eq 0 ]]; then
  MAX_KV_CACHE_CAPACITY="$(round_up $(( GOLDEN_SEQ_LEN + MAX_GEN )) 128)"
  (( MAX_KV_CACHE_CAPACITY < 256 )) && MAX_KV_CACHE_CAPACITY=256
fi
echo "[few-layer] engine bounds: maxInputLen=${MAX_INPUT_LEN} maxKVCacheCapacity=${MAX_KV_CACHE_CAPACITY} (prompt ${GOLDEN_SEQ_LEN} tokens)"

FORCED_TOKENS="${WORKDIR}/forced_tokens.txt"
"${PYBIN}" - "${GOLDEN}" "${FORCED_TOKENS}" <<'PY'
import sys
from safetensors import safe_open
with safe_open(sys.argv[1], framework="pt") as f:
    rows = f.get_tensor("generated_ids").tolist()
with open(sys.argv[2], "w") as out:
    out.write("\n".join(" ".join(str(t) for t in row) for row in rows) + "\n")
PY
stage_end

# ---------------------------------------------------------------------------
# 2. EdgeLLM export (first N decoder layers)
# ---------------------------------------------------------------------------
stage_begin "2/5 EdgeLLM export"
# --skip-visual / --skip-audio: only the LLM backbone is compared, so a VLM/omni
# checkpoint's encoder towers are pure cost here. Both are no-ops for a text-only model.
TARGET_ARGS=()
if [[ -n "${TARGET_SM}" ]]; then
  TARGET_ARGS=(--target-sm "${TARGET_SM}")
  if (( TARGET_SM >= 120 )); then
    export EDGELLM_NVFP4_MOE_TARGET=sm12x
  elif (( TARGET_SM == 110 )); then
    export EDGELLM_NVFP4_MOE_TARGET=sm110
  elif (( TARGET_SM >= 100 )); then
    export EDGELLM_NVFP4_MOE_TARGET=sm100
  fi
fi
"${PYBIN}" -m tensorrt_edgellm.scripts.export \
  "${MODEL}" "${EXPORT_DIR}" \
  --num-decoder-layer "${NUM_LAYERS}" \
  --skip-visual --skip-audio \
  ${TARGET_ARGS[@]+"${TARGET_ARGS[@]}"} \
  ${WEIGHT_ONLY_ARGS[@]+"${WEIGHT_ONLY_ARGS[@]}"} \
  ${MTP_EXPORT_ARGS[@]+"${MTP_EXPORT_ARGS[@]}"}
stage_end

# A sliding-window model (e.g. Gemma4) can bake a per-layer kv_cache_capacity into the
# export that exceeds the prompt-derived bound above -- llmBuilder.cpp rejects any layer
# whose capacity is greater than --maxKVCacheCapacity. Raise the bound to cover it rather
# than truncating the model's native window, unless the caller explicitly pinned a value.
if [[ "${MAX_KV_CACHE_CAPACITY_OVERRIDDEN}" -eq 0 ]]; then
  NATIVE_KV_CACHE_CAPACITY="$("${PYBIN}" - "${EXPORT_DIR}/llm/config.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    cfg = json.load(f)
layers = cfg.get("kv_layer_configs") or []
capacities = [lc.get("kv_cache_capacity", 0) for lc in layers if lc]
print(max(capacities, default=0))
PY
  )"
  if (( NATIVE_KV_CACHE_CAPACITY >= MAX_KV_CACHE_CAPACITY )); then
    # A page of headroom beyond the native capacity, not just up to it: the exported graph
    # bakes in swa_kv_page_table / swa_kv_cache_mode inputs whenever any layer's capacity is
    # bounded, and llmBuilder.cpp requires those markers to agree with the engine actually
    # having a *reduced* (strictly smaller than max) pool -- an exact match reads as "no
    # reduction" and the graph/builder markers disagree.
    MAX_KV_CACHE_CAPACITY="$(( $(round_up "${NATIVE_KV_CACHE_CAPACITY}" 128) + 128 ))"
    echo "[few-layer] raised maxKVCacheCapacity to ${MAX_KV_CACHE_CAPACITY} to fit the" \
      "export's native per-layer kv_cache_capacity=${NATIVE_KV_CACHE_CAPACITY}"
  fi
fi

# ---------------------------------------------------------------------------
# 3. EdgeLLM build
# ---------------------------------------------------------------------------
# maxInputLen / maxKVCacheCapacity default tight (--max-input-len / --max-kv-cache-capacity):
# this validation uses short, fixed prompts (a few dozen tokens) and the dumper now copies the
# full KV cache ([..., maxSeqLen, ...]) rather than truncating to the valid length, so maxSeqLen
# == maxKVCacheCapacity directly drives the dump size. A small cap keeps the dump at a few hundred
# MB instead of multiple GB; the comparison slices each sequence to its real length in PyTorch.
# Engine build parameters can affect accuracy, so they are overridable for debugging.
stage_begin "3/5 EdgeLLM build"
EDGELLM_PLUGIN_PATH="${PLUGIN}" "${LLM_BUILD}" \
  --onnxDir "${EXPORT_DIR}/llm" \
  --engineDir "${ENGINE_DIR}" \
  --maxBatchSize "${BATCH_SIZE}" \
  --maxInputLen "${MAX_INPUT_LEN}" \
  --maxKVCacheCapacity "${MAX_KV_CACHE_CAPACITY}" \
  ${MTP_BASE_BUILD_ARGS[@]+"${MTP_BASE_BUILD_ARGS[@]}"}

# The MTP draft engine goes into the same engine dir as the base (llm_build writes
# spec_draft.engine / draft_config.json alongside spec_base.engine).
if [[ "${MTP}" -eq 1 ]]; then
  EDGELLM_PLUGIN_PATH="${PLUGIN}" "${LLM_BUILD}" \
    --onnxDir "${EXPORT_DIR}/mtp_draft" \
    --engineDir "${ENGINE_DIR}" \
    --maxBatchSize "${BATCH_SIZE}" \
    --maxInputLen "${MAX_INPUT_LEN}" \
    --maxKVCacheCapacity "${MAX_KV_CACHE_CAPACITY}" \
    --maxDraftTreeSize "${SPEC_VERIFY_SIZE}" \
    --specDraft
fi
stage_end

# ---------------------------------------------------------------------------
# 4. EdgeLLM inference with the per-layer dump (greedy via top_k=1 in the JSON,
#    fixed length via EDGELLM_IGNORE_EOS).
# ---------------------------------------------------------------------------
stage_begin "4/5 EdgeLLM inference"
EDGELLM_PLUGIN_PATH="${PLUGIN}" \
EDGELLM_IGNORE_EOS=1 \
EDGELLM_FORCE_TOKENS_FILE="${FORCED_TOKENS}" \
EDGELLM_DUMP_LOGITS_KVCACHE_LAYERS="${NUM_LAYERS}" \
EDGELLM_DUMP_LOGITS_KVCACHE_DIR="${DUMP_DIR}" \
"${LLM_INFER}" \
  --engineDir "${ENGINE_DIR}" \
  --inputFile "${INPUT_FILE}" \
  --outputFile "${WORKDIR}/out.json" \
  --maxGenerateLength "${MAX_GEN}" \
  ${MTP_INFER_ARGS[@]+"${MTP_INFER_ARGS[@]}"} \
  ${REUSE_INFER_ARGS[@]+"${REUSE_INFER_ARGS[@]}"}
stage_end

# ---------------------------------------------------------------------------
# 5. Compare
# ---------------------------------------------------------------------------
stage_begin "5/5 compare"
# Pass --verbose through to the comparison only when requested on this script.
COMPARE_VERBOSE=()
[[ "${VERBOSE}" -eq 1 ]] && COMPARE_VERBOSE=(--verbose)
# Wrap in `if` so `set -e` does not abort before we capture the comparison's exit code.
if "${PYBIN}" "${REPO}/tests/compare_layer_dumps.py" \
  --golden "${GOLDEN}" \
  --edgellm "${DUMP_DIR}"/edgellm_dump*.safetensors \
  --teacher-forced \
  "${COMPARE_VERBOSE[@]}" \
  --cos "${COS}" --scale-tol "${SCALE_TOL}" --atol "${ATOL}" --rtol "${RTOL}"; then
  RC=0
else
  RC=$?
fi
stage_end

print_timing

DECODING=$([[ "${MTP}" -eq 1 ]] && echo "mtp" || echo "vanilla")
[[ "${CONTEXT_REUSE}" -eq 1 ]] && DECODING="${DECODING}+reuse"
echo
if [[ "${RC}" -eq 0 ]]; then
  echo "[few-layer] RESULT: PASS (model=${MODEL}, layers=${NUM_LAYERS}, decoding=${DECODING})"
else
  echo "[few-layer] RESULT: FAIL (model=${MODEL}, layers=${NUM_LAYERS}, decoding=${DECODING})"
fi
exit "${RC}"
