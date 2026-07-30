# Edge-LLM Qwen3.6-27B on Jetson AGX Thor — end-to-end MLPerf pipeline

Reproducible, precise steps: **build Edge-LLM → quantize (NVFP4 + NVFP4 lm_head +
FP8 KV, with MTP) → build engines → launch the server → run the MLPerf
Edge-Agentic benchmark.** Every command was run on-device. Helper files in this
folder: [`make_calibration.py`](make_calibration.py),
[`serve_edgellm.sh`](serve_edgellm.sh), [`config.yaml`](config.yaml),
[`chat_template_noreason.jinja`](chat_template_noreason.jinja).

## 0. Environment

Jetson AGX Thor, JetPack 7.2, CUDA 13.2, TensorRT 10.16, GPU `sm_110`, 128 GB
unified memory. Set once per shell (`<SCRATCH>` = your model-storage mount):

```bash
export REPO=$(git rev-parse --show-toplevel)          # this tensorrt-edge-llm checkout (branch feat/qwen36-27b-nvfp4-mlperf)
export VENV=/path/to/venv-edgellm-export              # python3.12 venv: pip install "$REPO[tools]"  (modelopt, transformers, torch, nvidia-cutlass-dsl, pybind11, onnx)
export WORK=$REPO/mlperf/artifacts                    # scratch for calib/onnx/engines/results
# Preferred: open-source NVFP4+MTP checkpoint (skip §2a/2b quantize).
#   https://huggingface.co/centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf
# Or unquantized Qwen/Qwen3.6-27B and quantize on device (§2a/2b).
export BASE_MODEL=<SCRATCH>/Qwen3.6-27B-NVFP4-W4A4-mlpinf
export CUDACXX=/usr/local/cuda/bin/nvcc; export PATH=/usr/local/cuda/bin:$PATH
mkdir -p "$WORK"
sudo nvpmodel -m 0 && sudo jetson_clocks              # lock clocks to max (stable timing)
```

If python3.12-dev headers are unavailable (no sudo): `apt-get download
python3.12-dev libpython3.12-dev`, `dpkg-deb -x <deb> $WORK/pyhdr`, and point the
pybind build at `$WORK/pyhdr/usr/include/python3.12` (see §1c).

## 1. Build Edge-LLM

```bash
cd "$REPO"
git submodule update --init --recursive

# 1a. CuTe DSL kernels from source (all variants for sm_110)
$VENV/bin/python kernelSrcs/build_cutedsl.py --kernels ALL --gpu_arch sm_110 --arch aarch64 -j 8
#   -> cpp/kernels/cuteDSLArtifact/aarch64/sm_110/libcutedsl_aarch64.a
#   (alternatively stage a prebuilt tarball into kernelSrcs/cuteDSLPrebuilt/)

# 1b. C++ runtime + plugin
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-thor -DCUDA_CTK_VERSION=13.2 -DENABLE_CUTE_DSL=ALL
make -j"$(nproc)"                 # -> build/libNvInfer_edgellm_plugin.so, build/examples/llm/llm_build
cd "$REPO"

# 1c. Python bindings (_edgellm_runtime) — required to serve
cd experimental/pybind && mkdir -p build && cd build
cmake .. -DTRT_PACKAGE_DIR=/usr -DEDGELLM_BUILD_DIR="$REPO/build" \
  -DCUDA_CTK_VERSION=13.2 -DCUDA_DIR=/usr/local/cuda \
  -DCMAKE_PREFIX_PATH="$($VENV/bin/python -c 'import pybind11;print(pybind11.get_cmake_dir())')" \
  -DPython_EXECUTABLE="$VENV/bin/python" \
  -DPython_INCLUDE_DIR="$WORK/pyhdr/usr/include/python3.12" \
  -DPython_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.12.so.1.0
make -j                           # -> _edgellm_runtime.cpython-312-*.so
cd "$REPO"
```

## 2. Quantize + export + build engines

Run the in-tree modules (from `$REPO`) so the current `tensorrt_edgellm` is used,
not an older installed copy.

```bash
# If BASE_MODEL is already centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf, skip 2a/2b and set QUANT="$BASE_MODEL".

# 2a. Calibration — 10% of the BFCL accuracy data as {messages, tools} JSONL.
#     Needs the BFCL v4 parquet (materialized once by the harness in §4, or pass its path).
$VENV/bin/python mlperf/make_calibration.py "$WORK/bfcl_calib.jsonl" <bfcl_v4_single_turn.parquet>

# 2b. Quantize: NVFP4 backbone + NVFP4 lm_head + FP8 KV. MTP draft auto-detected + quantized.
$VENV/bin/python -m tensorrt_edgellm.scripts.quantize llm \
  --model_dir "$BASE_MODEL" --output_dir "$WORK/quant-nvfp4" \
  --quantization nvfp4 --lm_head_quantization nvfp4 --kv_cache_quantization fp8 \
  --dataset "$WORK/bfcl_calib.jsonl" --num_samples 364
QUANT="$WORK/quant-nvfp4"

# 2c. Export ONNX WITH tree-MTP bindings. --mtp-tree-base is a hidden flag (implies --mtp)
#     and adds tree_parent_ids / tree_depths; plain --mtp is chain-only.
$VENV/bin/python -m tensorrt_edgellm.scripts.export \
  "$QUANT" "$WORK/onnx" --mtp-tree-base --skip-visual
$VENV/bin/python -c "import onnx;g=onnx.load('$WORK/onnx/llm/model.onnx',load_external_data=False).graph;print([i.name for i in g.input if 'tree' in i.name])"
#   -> ['tree_parent_ids', 'tree_depths']

# 2d. Build base + draft engines (tree MTP; draft --maxInputLen MUST equal base's).
# Prefill max=8192 with opt=1024 (classic 2-profile). Keep maxKVCacheCapacity=32768 so
# multi-turn agentic context reuse does not hit "Insufficient KV cache capacity" /
# "Context cache lease exceeds the page-table row capacity" (seen with KV=16384).
B="$REPO/build"; export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$B:$B/cpp:$LD_LIBRARY_PATH
export EDGELLM_PLUGIN_PATH="$B/libNvInfer_edgellm_plugin.so"
$B/examples/llm/llm_build --onnxDir "$WORK/onnx/llm" --engineDir "$WORK/engines/base" \
  --specBase  --maxInputLen 8192 --optInputLen 1024 --maxKVCacheCapacity 32768 --maxBatchSize 1 \
  --maxKVPoolPages 1024 --maxVerifyTreeSize 32
$B/examples/llm/llm_build --onnxDir "$WORK/onnx/mtp_draft" --engineDir "$WORK/engines/draft" \
  --specDraft --maxInputLen 8192 --optInputLen 1024 --maxKVCacheCapacity 32768 --maxBatchSize 1 \
  --maxKVPoolPages 512 --maxDraftTreeSize 32

# 2e. Combine into ONE engine dir + reasoning-off chat template (required for tool calls).
cp "$WORK/engines/draft/spec_draft.engine" "$WORK/engines/base/"
cp "$WORK/engines/draft/draft_config.json" "$WORK/engines/base/"
cp mlperf/chat_template_noreason.jinja      "$WORK/engines/base/chat_template.jinja"
```

## 3. Launch the server (tree-MTP + context reuse)

`serve_edgellm.sh` reads `$REPO`, `$VENV`, `$WORK`. It serves the combined engine
dir with tree MTP (top-k 8 / step 6 / verify 32) and prefill-state-only context reuse.

```bash
bash mlperf/serve_edgellm.sh                # serves on :8001
curl -s http://localhost:8001/v1/models     # -> {"data":[{"id":"base",...}]}
```

Startup log should show `Speculative decoding enabled (top_k=8, step=6, tree=32)`
and `Context cache hybrid snapshots: recurrent=1 slots`.

## 4. Run the MLPerf Edge-Agentic benchmark

Two datasets / two metrics: **performance** = multi-turn agentic-coding replay
(1007 turns, scored by inline IoU of tool calls); **accuracy** = BFCL v4
single-turn function-calling gate (995 samples, AST match vs. ground truth). Both
deterministic (`temperature 0`, `seed 42`), reasoning off, single-stream.

```bash
git clone https://github.com/mlcommons/endpoints && cd endpoints
python3.12 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev,bfcl]"
# edit endpoint/model/tokenizer paths in $REPO/mlperf/config.yaml, then from the harness root:
inference-endpoint benchmark from-config --config $REPO/mlperf/config.yaml                  # both phases
inference-endpoint benchmark from-config --config $REPO/mlperf/config.yaml --mode perf      # performance only
inference-endpoint benchmark from-config --config $REPO/mlperf/config.yaml --accuracy-only  # accuracy only
```

Outputs under `report_dir`: `scores.json` (perf inline IoU + turn validity),
`accuracy/accuracy_results.json` (BFCL score), `report.txt` (Duration / TPS).
The performance dataset ships with the harness
(`examples/11_Edge_Agentic_Example/agentic_coding_2.5h.jsonl`, 2014 rows / 20
conversations); BFCL self-downloads on first run and drops its parquet into
`dataset_cache/bfcl_v4/` (feed that to `make_calibration.py` in §2a).

## Gotchas (each is a hard error otherwise)
- **Tree MTP** (`--draft-top-k > 1`) needs `--mtp-tree-base` at export (adds
  `tree_parent_ids`/`tree_depths`). Plain `--mtp` is **chain-only** — then
  `--draft-top-k 1` and `--verify-tree-size = draft-step + 1`.
- The **draft** engine's `--maxInputLen` must equal the **base**'s, or long-context
  prefill fails `satisfyProfile` on `hidden_states_from_draft`.
- **`maxKVCacheCapacity` must stay at 32768** for MLPerf agentic multi-turn reuse.
  Prefill `maxInputLen=8192` alone is not enough: with `KV=16384`, late turns hit
  `Insufficient KV cache capacity` / page-table lease errors and IoU collapses.
- **Both engines in one `--engine-dir`**; do not also pass a separate draft-dir flag.
- MTP + reuse at concurrency 1 needs `--recurrent-capture-interval 0` (endpoint-only
  capture, the opposite of non-MTP reuse) and a recurrent pool sized to **one slot**
  (~160 MB) — a multi-GiB pool errors "requires exactly one recurrent snapshot slot".
- The engine dir needs a `chat_template.jinja`; use a reasoning-off template so the
  model does not emit `<think>` spans (the harness runs reasoning off).
