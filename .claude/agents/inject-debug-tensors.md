---
name: inject-debug-tensors
description: >
  Temporarily injects TensorRT IDebugListener tensor-dump infrastructure
  into the working tree, builds the C++ runtime, runs a model FP16 accuracy
  check via compare_hf_trt_tensors.py, reports the verdict, and then
  automatically reverts all injected code changes.
  Requires TRT >= 10.12 (IDebugListener API).  No permanent source changes.
  Trigger: any request to "run debug injection", "verify tensor accuracy",
  "check model numerical accuracy against HF", or "/inject-debug-tensors".
allowed-tools:
  - Bash
  - Read
  - AskUserQuestion
---

# inject-debug-tensors Agent

Injects debug-tensor infrastructure from `.claude/agents/inject-debug-tensors/`,
runs an HF-vs-TRT accuracy check, then reverts every file it touched.

**Parameters** (parse from `$ARGUMENTS`):
- `TRT_PACKAGE_DIR` — required; if omitted use `$TRT_PACKAGE_DIR` env var, then `/usr/local/tensorrt`, then abort.
- `model` — required; HF model ID or local path.
- `python_env` — optional; venv path to activate. If omitted, detect available packages; create `<workspace>/venv` if missing.
- `prompt` — optional; default `"What is the capital of France?"`
- `workspace` — optional; default `/tmp/edgellm_debug_inject`

---

## Pre-flight — Commit check

Before doing anything else:

1. Run `git status --short` to check for uncommitted changes.
2. If the working tree is dirty, use **AskUserQuestion** to tell the user:
   > "You have uncommitted changes (listed above). Please commit or stash them
   > before running this agent — injection modifies source files and a clean tree
   > is required to safely revert afterwards. Reply 'done' once committed, or
   > 'abort' to cancel."
3. If the user replies 'abort' (or equivalent), stop immediately.
4. If the user replies 'done', re-run `git status --short`. If still dirty, ask again or abort.
5. Only proceed to Step 0 once `git diff --quiet && git diff --cached --quiet` passes.

---

## Step 0 — Setup

```bash
{ [ -d "tensorrt_edgellm" ] || [ -d "experimental" ]; } || { echo "ERROR: not in repo root"; exit 1; }
REPO_ROOT=$(pwd)
AGENT_DIR="$REPO_ROOT/.claude/agents/inject-debug-tensors"

# Resolve TRT_PACKAGE_DIR
TRT_PACKAGE_DIR="${TRT_ARG:-${TRT_PACKAGE_DIR}}"
[ -z "$TRT_PACKAGE_DIR" ] && [ -f "/usr/local/tensorrt/include/NvInferVersion.h" ] && TRT_PACKAGE_DIR="/usr/local/tensorrt"
[ -z "$TRT_PACKAGE_DIR" ] && { echo "ERROR: TRT_PACKAGE_DIR not set. Pass it as first argument."; exit 1; }
export TRT_PACKAGE_DIR
export LD_LIBRARY_PATH="$TRT_PACKAGE_DIR/lib:$LD_LIBRARY_PATH"

# Check TRT >= 10.12
HEADER="$TRT_PACKAGE_DIR/include/NvInferVersion.h"
[ -f "$HEADER" ] || { echo "ERROR: $HEADER not found"; exit 1; }
MAJOR=$(grep "^#define NV_TENSORRT_MAJOR" "$HEADER" | awk '{print $3}')
MINOR=$(grep "^#define NV_TENSORRT_MINOR" "$HEADER" | awk '{print $3}')
{ [ "$MAJOR" -gt 10 ] || { [ "$MAJOR" -eq 10 ] && [ "$MINOR" -ge 12 ]; }; } || {
    echo "ERROR: TRT $MAJOR.$MINOR < 10.12 — IDebugListener not available"; exit 1; }
echo "[OK] TRT $MAJOR.$MINOR, IDebugListener supported"

```

---

## Step 1 — Inject

```bash
# New files
mkdir -p "$REPO_ROOT/cpp/debug" "$REPO_ROOT/examples/debug"
cp "$AGENT_DIR/debugTensorDumper.h"        "$REPO_ROOT/cpp/debug/"
cp "$AGENT_DIR/debugTensorDumper.cpp"      "$REPO_ROOT/cpp/debug/"
cp "$AGENT_DIR/compare_hf_trt_tensors.py"  "$REPO_ROOT/examples/debug/"
cp "$AGENT_DIR/debug_README.md"            "$REPO_ROOT/examples/debug/README.md"

# Patch existing files
patch -p1 --forward < "$AGENT_DIR/existing_files.patch" || {
    echo "ERROR: patch failed — rolling back"; rm -rf "$REPO_ROOT/cpp/debug" "$REPO_ROOT/examples/debug"; exit 1; }
echo "[Inject] Done"
```

---

## Step 2 — Build

Skip if `build/examples/llm/llm_build` exists and is newer than `cpp/debug/debugTensorDumper.cpp`.

```bash
if [ -x "$REPO_ROOT/build/examples/llm/llm_build" ] && \
   [ "$REPO_ROOT/build/examples/llm/llm_build" -nt "$REPO_ROOT/cpp/debug/debugTensorDumper.cpp" ]; then
    echo "[Build] Skipping — already up to date"
else
    git submodule update --init --quiet
    mkdir -p "$REPO_ROOT/build" && cd "$REPO_ROOT/build"
    cmake .. -DTRT_PACKAGE_DIR="$TRT_PACKAGE_DIR" -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
    make -j$(nproc) 2>&1 | tail -20
    [ -x "$REPO_ROOT/build/examples/llm/llm_build" ] || { echo "ERROR: build failed"; exit 1; }
    cd "$REPO_ROOT"
fi
```

---

## Step 3 — Python environment

```bash
if [ -n "$PYTHON_ENV" ]; then
    source "$PYTHON_ENV/bin/activate"
elif python3 -c "import torch, transformers, safetensors" 2>/dev/null; then
    : # packages already available
else
    VENV_PATH="$WORKSPACE/venv"
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install --quiet torch transformers safetensors numpy Pillow
    echo "[Venv] Created at $VENV_PATH"
fi
```

---

## Step 4 — Export ONNX (skip if exists)

```bash
MODEL_NAME=$(basename "$MODEL")
ONNX_DIR="$WORKSPACE/$MODEL_NAME/onnx"
if [ -f "$ONNX_DIR/model.onnx" ]; then
    echo "[Export] Skipping — already exists"
else
    tensorrt-edgellm-export-llm --model_dir "$MODEL" --output_dir "$ONNX_DIR" --device cuda \
        || { FAILED=export; }
fi
```

---

## Step 5 — Build TRT engine with --debugTensors (skip if exists)

```bash
ENGINE_DIR="$WORKSPACE/$MODEL_NAME/engine_debug"
if python3 -c "import json; assert json.load(open('$ENGINE_DIR/config.json')).get('debug_tensors')" 2>/dev/null; then
    echo "[Engine] Skipping — already built"
elif [ -z "$FAILED" ]; then
    "$REPO_ROOT/build/examples/llm/llm_build" \
        --onnxDir "$ONNX_DIR" --engineDir "$ENGINE_DIR" \
        --maxBatchSize 4 --maxInputLen 1024 --maxKVCacheCapacity 4096 \
        --debugTensors || { FAILED=engine; }
fi
```

---

## Step 6 — Inference with --dumpTensors

```bash
DUMP_DIR="$WORKSPACE/$MODEL_NAME/trt_tensors"
if [ -z "$FAILED" ]; then
    mkdir -p "$DUMP_DIR"
    python3 -c "
import json, os
data = {'batch_size':1,'max_generate_length':8,'apply_chat_template':True,
        'requests':[{'messages':[{'role':'user','content':os.environ.get('PROMPT','What is the capital of France?')}]}]}
json.dump(data, open('$WORKSPACE/input.json','w'))
"
    "$REPO_ROOT/build/examples/llm/llm_inference" \
        --engineDir "$ENGINE_DIR" \
        --inputFile "$WORKSPACE/input.json" \
        --outputFile "$WORKSPACE/trt_output.json" \
        --dumpTensors "$DUMP_DIR" || { FAILED=inference; }
    [ -z "$FAILED" ] && cat "$WORKSPACE/trt_output.json"
fi
```

---

## Step 7 — Compare HF vs TRT

```bash
[ -z "$FAILED" ] && python3 "$REPO_ROOT/examples/debug/compare_hf_trt_tensors.py" \
    --model-dir "$MODEL" --trt-dump "$DUMP_DIR" \
    --prompt "${PROMPT:-What is the capital of France?}" \
    --dtype fp16 --max-new-tokens 8
```

Report verdict: **PASS** (cos ≥ 0.999 mean + text match) / **WARN** (0.95–0.999) / **FAIL** (< 0.95 or text diverges).

---

## Step 8 — Revert (ALWAYS run)

Restores exactly the files that were modified — no broad resets.

```bash
cd "$REPO_ROOT"

# Restore patched files to their HEAD state (only touches the 19 known files)
git checkout HEAD -- \
    cpp/CMakeLists.txt \
    cpp/common/trtUtils.h \
    cpp/builder/llmBuilder.h \
    cpp/builder/llmBuilder.cpp \
    cpp/builder/visualBuilder.h \
    cpp/builder/visualBuilder.cpp \
    cpp/runtime/llmEngineRunner.h \
    cpp/runtime/llmEngineRunner.cpp \
    cpp/runtime/llmInferenceRuntime.h \
    cpp/multimodal/common/multimodalRunner.h \
    cpp/multimodal/common/multimodalRunner.cpp \
    cpp/multimodal/qwen3_omni/audioRunner.h \
    cpp/multimodal/qwen3_omni/audioRunner.cpp \
    cpp/multimodal/internvl/internViTRunner.cpp \
    cpp/multimodal/qwen2/qwenViTRunner.cpp \
    cpp/multimodal/phi4mm/phi4mmViTRunner.cpp \
    examples/llm/llm_build.cpp \
    examples/llm/llm_inference.cpp \
    examples/multimodal/visual_build.cpp

# Remove injected directories
rm -rf "$REPO_ROOT/cpp/debug" "$REPO_ROOT/examples/debug"

git diff --quiet && git diff --cached --quiet \
    && echo "[Revert] ✓ Clean" \
    || { echo "[Revert] ⚠ Unexpected changes remain:"; git status --short; }
```
