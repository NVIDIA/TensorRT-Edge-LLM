#!/bin/bash
# Serve the combined base+draft engine dir with MTP and context reuse on :8001.
# Requires: REPO (edge-llm repo root), VENV (python venv), WORK (artifacts dir).
set -e
: "${REPO:?set REPO to the tensorrt-edge-llm repo root}"
: "${VENV:?set VENV to the python venv}"
: "${WORK:?set WORK to the artifacts dir}"
export PATH=/usr/local/cuda/bin:$PATH
B="$REPO/build"
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:$B:$B/cpp:$LD_LIBRARY_PATH"
export EDGELLM_PLUGIN_PATH="$B/libNvInfer_edgellm_plugin.so"
export PYTHONPATH="$REPO:$B/pybind:$PYTHONPATH"
cd "$REPO"
exec "$VENV/bin/python" -m experimental.server \
  --engine-dir "$WORK/engines/base" \
  --draft-top-k 2 --draft-step 8 --verify-tree-size 16 \
  --enable-context-reuse --context-reuse-prefill-state-only \
  --context-cache-max-records 1024 \
  --context-cache-recurrent-snapshot-pool-bytes 160000000 \
  --context-cache-partial-kv-snapshot-pool-bytes 536870912 \
  --recurrent-capture-interval 0 \
  --port 8001
