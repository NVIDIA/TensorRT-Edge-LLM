# Source Injection for Tensor Debugging

This guide describes the `IDebugListener` instrumentation path after a
few-layer test identifies a mismatch. The bundled patch is revision-specific:
check it against the current builder and runtime in an isolated worktree
before applying it. Keep the provider's PyTorch run as the reference.

The steps use the assets in `skills/edge-llm-accuracy-debugging/` to build an
instrumented runtime and compare its output against the PyTorch reference.
The build and inference flags require that instrumentation; verify them before
running the example commands.

**Inputs:**
- `TRT_PACKAGE_DIR` — required; if omitted use `$TRT_PACKAGE_DIR` env var, then `/usr/local/tensorrt`, then abort.
- `model` — required; HF model ID or local path.
- `python_env` — optional; venv path to activate. If omitted, detect available packages; create `<workspace>/venv` if missing.
- `prompt` — optional; default `"What is the capital of France?"`
- `workspace` — optional; default `/tmp/edgellm_debug_inject`

---

## Pre-flight — Commit check

Before doing anything else:

1. Run `git status --short` to check for uncommitted changes.
2. If the working tree is dirty, use a separate clean worktree for
   instrumentation. Do not modify or reset unrelated changes.
3. Only proceed once the diagnostic worktree is clean and the patch applies to
   the current revision. Preserve results before removing that worktree.

---

## Step 0 — Setup

```bash
{ [ -d "tensorrt_edgellm" ] || [ -d "experimental" ]; } || { echo "ERROR: not in repo root"; exit 1; }
REPO_ROOT=$(pwd)
AGENT_DIR="$REPO_ROOT/skills/edge-llm-accuracy-debugging"

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
git apply --check "$AGENT_DIR/references/source-injection.patch" || exit 1

# New files
mkdir "$REPO_ROOT/cpp/debug" "$REPO_ROOT/examples/debug" || exit 1
cp "$AGENT_DIR/assets/debugTensorDumper.h" "$REPO_ROOT/cpp/debug/" || exit 1
cp "$AGENT_DIR/assets/debugTensorDumper.cpp" "$REPO_ROOT/cpp/debug/" || exit 1
cp "$AGENT_DIR/scripts/compare_hf_trt_tensors.py" "$REPO_ROOT/examples/debug/" || exit 1
cp "$AGENT_DIR/references/full-intermediate-tensors.md" "$REPO_ROOT/examples/debug/README.md" || exit 1

# Patch existing files
git apply "$AGENT_DIR/references/source-injection.patch" || exit 1
echo "[Inject] Done"
```

If either check fails, preserve the diagnostic worktree for inspection. The
destination directories must be new; do not overwrite or automatically delete
existing debug files when recovering from a failed or interrupted injection.

---

## Step 2 — Build

Skip if `build/bin/llm_build` exists and is newer than `cpp/debug/debugTensorDumper.cpp`.

```bash
if [ -x "$REPO_ROOT/build/bin/llm_build" ] && \
   [ "$REPO_ROOT/build/bin/llm_build" -nt "$REPO_ROOT/cpp/debug/debugTensorDumper.cpp" ]; then
    echo "[Build] Skipping — already up to date"
else
    git submodule update --init --quiet
    mkdir -p "$REPO_ROOT/build" && cd "$REPO_ROOT/build"
    cmake .. -DTRT_PACKAGE_DIR="$TRT_PACKAGE_DIR" -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
    make -j$(nproc) 2>&1 | tail -20
    [ -x "$REPO_ROOT/build/bin/llm_build" ] || { echo "ERROR: build failed"; exit 1; }
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
    trtedge export "$MODEL" "$ONNX_DIR" \
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
    "$REPO_ROOT/build/bin/llm_build" \
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
    "$REPO_ROOT/build/bin/llm_inference" \
        --engineDir "$ENGINE_DIR" \
        --inputFile "$WORKSPACE/input.json" \
        --outputFile "$WORKSPACE/trt_output.json" \
        --dumpTensors "$DUMP_DIR" || { FAILED=inference; }
    [ -z "$FAILED" ] && cat "$WORKSPACE/trt_output.json"
fi
```

---

## Step 7 — Compare HF vs TRT

Use the same checkpoint snapshot as the engine export. A Hub ID requires
`--revision` with its full immutable revision; local checkpoint directories
are loaded offline. Use a fresh workspace when changing checkpoints rather
than reusing ONNX files or engines built from different weights.

```bash
[ -z "$FAILED" ] && python3 "$REPO_ROOT/examples/debug/compare_hf_trt_tensors.py" \
    --model-dir "$MODEL" --trt-dump "$DUMP_DIR" \
    --prompt "${PROMPT:-What is the capital of France?}" \
    --dtype fp16 --max-new-tokens 8
```

Report verdict: **PASS** (cos ≥ 0.999 mean + text match) / **WARN** (0.95–0.999) / **FAIL** (< 0.95 or text diverges).

---

## Step 8 — Clean up

Keep the instrumented build and its dumps separate from production artifacts.
When finished, remove only the dedicated diagnostic worktree after checking
its status and preserving any results you need. Do not reset files in a
shared worktree.
