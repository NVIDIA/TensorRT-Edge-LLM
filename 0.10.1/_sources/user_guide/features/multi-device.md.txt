<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Multi-Device Inference

## Supported Parallelism

Edge-LLM currently supports tensor parallel inference with TP=2. Context
parallelism (CP), expert parallelism (EP), and pipeline parallelism (PP) are not supported yet.

## Supported Hardware

Supported TP=2 hardware includes a pair of NVIDIA DGX Spark systems, with one
GB10 GPU per system. Connect the systems through their high-speed ConnectX-7
QSFP/RoCE interfaces. The Dual DGX Spark path runs one MPI process per system
and uses NCCL for cross-system communication.

## Tensor Parallel Inference

Tensor parallelism (TP) splits supported model projections across multiple GPUs
and uses Edge-LLM AllReduce plugins to combine rank-local results. The runtime uses NCCL for plugin communication and generation-state synchronization.

The workflow below uses the default local launch mode: one process owns both
GPUs and `RuntimeCoordinator` creates one worker thread and one
`LLMRankRuntime` per rank. Rank 0 owns generated text, output JSON, and profile
output. On Dual DGX Spark, MPI launches one process per system, and each
process owns one global TP rank on its node-local CUDA device 0. Both launch
modes use the same `RuntimeCoordinator` and rank-local runtime path.

Single-device callers continue to use the existing `LLMInferenceRuntime` API.
The runtime represents that case as a size-one parallel plan and executes its
single `LLMRankRuntime` inline on the caller thread. It does not create a worker
thread, collective resources, or cross-rank dispatch. Multi-device coordination
is activated only when the resolved parallel world contains more than one rank.

## Prerequisites

Follow the platform-specific [Installation Guide](../getting_started/installation.md)
and build Edge-LLM with multi-device support enabled. Add the following option
to the platform's normal CMake configuration:

```bash
-DENABLE_MULTI_DEVICE=ON
```

For example:

```bash
cmake -S . -B build \
  -DTRT_PACKAGE_DIR=/path/to/TensorRT \
  -DENABLE_MULTI_DEVICE=ON \
  -DENABLE_CUTE_DSL=ALL \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

`ENABLE_CUTE_DSL` must include the `gemm_nvfp4` group for the documented
NVFP4 TP2 path because it links the NVFP4 GEMM artifacts used by
`FusedNvfp4GemmAllReducePlugin`. `-DENABLE_CUTE_DSL=ALL` includes this group
and is recommended for customer builds because it enables every available
CuTe DSL kernel group, allowing both single-device and multi-device execution
to use the optimized kernels applicable to the selected model.

NCCL must be installed for the target platform and discoverable at runtime.
When it is not on the default loader path, set `EDGELLM_NCCL_SO_PATH` to the
platform's `libnccl.so.2` before starting inference.

The primary commands in this guide cover local threaded execution. For Dual
DGX Spark, also build with `-DENABLE_MULTI_DEVICE_MPI=ON` and install a
compatible MPI implementation. Both systems must expose the same engine and
checkpoint paths and allow non-interactive SSH for MPI process launch.
Configure MPI and NCCL to use the high-speed QSFP/RoCE interfaces rather than
the management network. Other multi-host targets require separate validation.

## Export Rank-Local ONNX Graphs

Export all TP ranks in one command:

```bash
export CHECKPOINT=/path/to/checkpoint
export ONNX_ROOT=/path/to/onnx/model-tp2

tensorrt-edgellm-export "$CHECKPOINT" "$ONNX_ROOT" \
  --tp-size 2 \
  --skip-visual \
  --skip-audio \
  --skip-code2wav
```

The LLM directory contains one graph per rank and one shared world config:

```text
llm/model_world2_rank0.onnx
llm/model_world2_rank1.onnx
llm/config_world2.json
```

`config_world2.json` contains a `rank_configs` entry for every rank. World size
is inferred from the number of entries; rank identity is not duplicated in the
shared `builder_config`.

## Build One Engine Per Rank

Build each rank with the same output directory. Rank 0 writes shared runtime
artifacts, and every rank writes its own engine:

```bash
export ONNX_DIR="$ONNX_ROOT/llm"
export ENGINE_DIR=/path/to/engines/model-tp2
export LLM_BUILD=./build/examples/llm/llm_build

mkdir -p "$ENGINE_DIR"

for RANK in 0 1; do
  "$LLM_BUILD" \
    --onnxDir="$ONNX_DIR" \
    --engineDir="$ENGINE_DIR" \
    --maxBatchSize=1 \
    --maxInputLen=2048 \
    --maxKVCacheCapacity=4096 \
    --tpSize=2 \
    --tpRank="$RANK"
done
```

On Dual DGX Spark, run one rank build per system against the same ONNX and
engine directories instead of running the loop on one system. Build global TP
rank 0 on the first system with `--tpRank=0 --localDevice=0`, and build global
TP rank 1 on the second system with `--tpRank=1 --localDevice=0`. Both systems
use CUDA device 0 because CUDA ordinals are node-local.

The completed engine directory contains:

```text
llm_world2_rank0.engine
llm_world2_rank1.engine
config_world2.json
tokenizer files
embedding files
```

Do not rename rank artifacts. The runtime resolves them from the shared world
config and validates that its `rank_configs` length matches `--tpSize`.

### Build Directly From The Checkpoint

The experimental ONNX-less builder can produce the same rank-local engine
layout without exporting ONNX. Invoke it once per rank with the same checkpoint
and output directory:

```bash
export CHECKPOINT=/path/to/checkpoint
export ENGINE_DIR=/path/to/engines/model-tp2
export PLUGIN_PATH=/path/to/build/libNvInfer_edgellm_plugin.so

mkdir -p "$ENGINE_DIR"

for RANK in 0 1; do
  .venv/bin/tensorrt-edgellm-build \
    --model-dir "$CHECKPOINT" \
    --engine-dir "$ENGINE_DIR" \
    --plugin-path "$PLUGIN_PATH" \
    --components llm \
    --max-input-len 2048 \
    --max-kv-cache-capacity 4096 \
    --max-batch-size 1 \
    --tp-size 2 \
    --tp-rank "$RANK"
done
```

For supported NVFP4 TP2 models, the direct builder retains TensorRT-native
weights in each rank engine. The fused `o_proj` and `down_proj` plugin weights
use rank-neutral checkpoint recipes instead: each runtime rank materializes
only its packed local shard, registers the immutable weight and block-scale
buffers with the plugin, and keeps them alive for the engine lifetime. These
resources are not TensorRT input bindings.

Tensor-parallel direct builds currently support the LLM component without
speculative decoding.

## Run TP2 Inference

Use the same request JSON accepted by single-device `llm_inference` and add
`--tpSize=2`:

```bash
export LLM_INFERENCE=./build/examples/llm/llm_inference

"$LLM_INFERENCE" \
  --engineDir="$ENGINE_DIR" \
  --inputFile=/path/to/input.json \
  --outputFile=/path/to/output.json \
  --tpSize=2
```

The process initializes both rank-local engines and NCCL resources before
executing requests. Only rank 0 writes responses and profiling artifacts.
For engines produced by the checkpoint-direct builder, pass the same
checkpoint so each rank can prepare its external weights during
initialization:

```bash
"$LLM_INFERENCE" \
  --engineDir="$ENGINE_DIR" \
  --checkpointDir="$CHECKPOINT" \
  --inputFile=/path/to/input.json \
  --outputFile=/path/to/output.json \
  --tpSize=2
```

### Run On Dual DGX Spark

Replace `spark-0` and `spark-1` with host names or addresses reachable through
the MPI launcher. Launch one process per system and export the runtime library
paths to both ranks:

```bash
mpirun \
  --host spark-0:1,spark-1:1 \
  -np 2 \
  --map-by ppr:1:node \
  --bind-to none \
  -x LD_LIBRARY_PATH \
  -x EDGELLM_PLUGIN_PATH \
  -x EDGELLM_NCCL_SO_PATH \
  "$LLM_INFERENCE" \
    --engineDir="$ENGINE_DIR" \
    --inputFile=/path/to/input.json \
    --outputFile=/path/to/output.json \
    --tpSize=2
```

For engines produced by the checkpoint-direct builder, also pass
`--checkpointDir="$CHECKPOINT"`. Verify from the NCCL startup log that the
selected network interfaces are the QSFP/RoCE links, not the management
interfaces.

To collect performance metrics:

```bash
"$LLM_INFERENCE" \
  --engineDir="$ENGINE_DIR" \
  --inputFile=/path/to/input.json \
  --outputFile=/path/to/output.json \
  --tpSize=2 \
  --warmup=3 \
  --dumpProfile \
  --profileOutputFile=/path/to/profile.json
```

Always compare generated output against the corresponding single-device run
before using performance results. Use the same checkpoint, request JSON,
builder limits, warmup count, and runtime environment for both measurements.

## Current Limitations

- TP2 is the initial supported topology.
- Tensor-parallel speculative decoding is not supported.
- Context reuse is not supported with TP greater than one.
- Qwen3-Omni audio output and Thinker-Talker streaming are not supported with
  TP greater than one.
- The supported cross-system topology is Dual DGX Spark. Other multi-host
  hardware requires separate validation.
