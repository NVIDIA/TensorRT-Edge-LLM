# Running the Edge-LLM OpenAI-Compatible Server

`experimental.server` wraps the C++ runtime in an OpenAI-compatible HTTP server
(`/v1/models`, `/v1/chat/completions` with optional SSE streaming and tool
calling). This README covers serving a **pre-built engine** and enabling
**KV-cache (context) reuse**, including the pool-sizing that hybrid
(linear-attention / GDN) models require.

## 1. Build steps

Commands below are for **Jetson AGX Thor** (aarch64, JetPack 7.2, CUDA 13.2,
TensorRT 10.16, `sm_110`). Adjust `EMBEDDED_TARGET` / `CUDA_CTK_VERSION` /
`CMAKE_CUDA_ARCHITECTURES` for other platforms. `export CUDACXX=/usr/local/cuda/bin/nvcc`
first (nvcc is often not on `PATH`).

### 1a. C++ runtime + plugin

```bash
git submodule update --init --recursive
# Stage the prebuilt CuTe DSL artifact for your arch so CMake can extract it
#   (or build from source: python kernelSrcs/build_cutedsl.py --gpu_arch sm_110 --arch aarch64)
mkdir -p kernelSrcs/cuteDSLPrebuilt
cp <path>/cutedsl_aarch64_sm_110_cuda13.tar.gz kernelSrcs/cuteDSLPrebuilt/

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-thor -DCUDA_CTK_VERSION=13.2 -DENABLE_CUTE_DSL=ALL
make -j$(nproc)     # → build/libNvInfer_edgellm_plugin.so, build/examples/llm/{llm_build,llm_inference}
cd ..
```

### 1b. Python bindings (`_edgellm_runtime`)

```bash
pip install -r requirements-server.txt          # fastapi, uvicorn, pybind11
cd experimental/pybind && mkdir -p build && cd build
cmake .. -DTRT_PACKAGE_DIR=/usr -DEDGELLM_BUILD_DIR=<repo>/build \
  -DCUDA_CTK_VERSION=13.2 -DCUDA_DIR=/usr/local/cuda \
  -DCMAKE_PREFIX_PATH=$(python -c "import pybind11;print(pybind11.get_cmake_dir())")
make -j          # → _edgellm_runtime.<abi>.so
cd ../../..
```
Standalone builds link the prebuilt CuTe DSL archive
(`cpp/kernels/cuteDSLArtifact/<arch>/<tag>/libcutedsl_*.a`) automatically.

### 1c. Export a checkpoint to ONNX (host or device, separate venv)

```bash
python3 -m venv ~/venv-export && source ~/venv-export/bin/activate
pip install ".[tools]" -r requirements-server.txt
# Pre-quantized checkpoints export directly; add --skip-visual for text-only VLMs
tensorrt-edgellm-export /path/to/HF-or-quantized-ckpt out/onnx --skip-visual
# → out/onnx/llm/{model.onnx, config.json, tokenizer.json, ...}
```

### 1d. Build the TensorRT engine (Paged KV)

```bash
export EDGELLM_PLUGIN_PATH=<repo>/build/libNvInfer_edgellm_plugin.so   # required, absolute path
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:<repo>/build:<repo>/build/cpp:$LD_LIBRARY_PATH
<repo>/build/examples/llm/llm_build \
  --onnxDir out/onnx/llm --engineDir out/engines \
  --maxInputLen 31744 --maxKVCacheCapacity 32768 --maxBatchSize 1 \
  --maxKVPoolPages 768      # page=128 tok; floor = maxBatchSize*ceil(maxKVCacheCapacity/128).
                            # Use well above the floor to RETAIN long prefixes for context reuse.
```

The engine dir gets `llm.engine` + tokenizer/config/chat-template, so it is
self-contained for serving.

## 2. Environment

```bash
B=<repo>/build
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$B:$B/cpp:$LD_LIBRARY_PATH
export EDGELLM_PLUGIN_PATH=$B/libNvInfer_edgellm_plugin.so
export PYTHONPATH=<repo>:$PYTHONPATH
```

## 3. Serve

From a HuggingFace checkpoint or local checkpoint (exports + builds, then serves):
```bash
python -m experimental.server --model Qwen/Qwen3-1.7B --port 8000
```

From a **pre-built engine directory** (serve as-is, no export/build):
```bash
python -m experimental.server --engine-dir /path/to/engines --port 8000
```

Verify: `curl -s http://localhost:8000/v1/models`.

`--model` and `--engine-dir` are mutually exclusive (provide exactly one).

## 4. KV-cache (context) reuse

Content-addressed reuse lets a later request that shares a prefix with an
earlier one **skip re-prefilling the shared prefix**. Enable it with:

```bash
python -m experimental.server \
  --engine-dir /path/to/engines \
  --enable-context-reuse \
  --context-cache-max-records 1024 \
  --context-cache-recurrent-snapshot-pool-bytes  12884901888  \
  --context-cache-partial-kv-snapshot-pool-bytes 536870912    \
  --recurrent-capture-interval 512 \
  --port 8000
```

On startup the server logs the realized slot counts:
```
Context cache hybrid snapshots: recurrent=83 slots (154927104 bytes/slot), partialKV=128 slots (...)
```

The multi-slot example above is for ordinary hybrid decoding. The current serialized Hybrid-MTP path retains one
complete predecessor endpoint and therefore uses a different configuration. For the currently validated
Qwen3.6-27B MTP engine:

```bash
python -m experimental.server \
  --engine-dir /path/to/complete/mtp-engine-bundle \
  --enable-context-reuse \
  --max-batch-size 1 \
  --context-cache-max-records 1 \
  --context-cache-recurrent-snapshot-pool-bytes 154927104 \
  --context-cache-partial-kv-snapshot-pool-bytes 8912896 \
  --context-reuse-prefill-state-only \
  --recurrent-capture-interval 0 \
  --draft-top-k 8 \
  --draft-step 6 \
  --verify-tree-size 60 \
  --port 8000
```

These byte counts are specific to that engine. Startup must report exactly one recurrent slot and at least one
bundled partial-KV slot. Hybrid-MTP reuse is currently text-only, batch-one, endpoint-only, and prefill-state-only.
The tool-aware server computes the required prompt replay tail from the active chat template; ordinary non-tool
prompts use MTP normally but do not publish a reusable Hybrid-MTP predecessor endpoint. A complete MTP bundle is
selected directly with `--engine-dir`; do not also pass `--spec-decode-engine-dir`.

### Sizing the pools (important for hybrid / GDN models)

For a hybrid model (some linear-attention/GDN layers), each reused prefix needs a
**recurrent-state snapshot**. The per-slot size is model-dependent and printed in
the startup log (`bytes/slot`). If the pool is smaller than one slot you get
**`recurrent=0 slots` and reuse silently never fires** — every request recomputes.

1. **Recurrent snapshot pool** ≥ `recurrent_bytes_per_slot × (retained prefixes)`.
   E.g. Qwen3.6-27B here uses ~148 MiB/slot; a 128 MiB pool → 0 slots (broken).
   Use several GiB (the example above → 83 slots).
2. When used, **`--recurrent-capture-interval`** must be a **multiple of 128**.
   It defaults to 0, which captures no intermediate recurrent snapshots; request
   endpoints are still captured. Set e.g. `512` when requests share an interior
   prefix rather than a previously published endpoint.
3. **KV pool pages** (`--maxKVPoolPages` at **engine build** time) must be large
   enough to *retain* the reused prefix. A prefix of `T` tokens needs
   `ceil(T/128)` pages retained *plus* headroom for the active request. If the
   pool is too small the runtime falls back to a cold (full-recompute) prefill.
   For long-context reuse build with a generous page count (e.g. `768` for 32K
   context), not just the `ceil(maxKVCacheCapacity/128)` floor.

### Quick self-check

Send the same long prompt (> 1 page = 128 tokens) twice; the second request's
time-to-first-token should drop sharply if reuse is active:
```bash
# run1 cold, run2 should be much faster when reuse works
```
(Prompts under one page are never published, so short prompts show no reuse.)

## 5. Streaming / TTFT note

For tool-calling requests the server buffers the full generation to parse tool
call arguments, then emits them. It now sends an initial `{"role":"assistant"}`
delta at first-token time so streaming clients still observe a meaningful TTFT.
For a pure prefill→first-token latency number, measure a **non-tool** streaming
request (incremental token deltas).

## 6. Sampling / tool options

See the request body fields in `api_server.py` and the tool-calling contract in
`tool_calling.py` (`tools`, `tool_choice`, `assistant.tool_calls`, `tool`
messages). Determinism: `temperature: 0` + `seed`.
