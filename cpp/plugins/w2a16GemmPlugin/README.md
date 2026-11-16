<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# W2A16 dense GEMM plugin

`W2A16GemmPlugin`, version `1`, namespace `""`, exposes the
[IQ2_XXS dense runner](../../kernels/gemm/w2a16_cutedsl/cuteDslW2A16GemmRunner.h)
as a TensorRT V3 layer on SM120 GPUs, GB10 (SM121, 48 SMs) and Thor (SM110). Enable the `w2a16_gemm`
CuTeDSL AOT group when building `NvInfer_edgellm_plugin`.

## Inputs and attributes

Set the required scalar INT32 fields `gemm_n=N` and `gemm_k=K`.
The shape limits are `1 <= M <= 32768`, `0 < N <= 4194240`, `N%8=0`,
and positive `K%256=0`. For batched activations, `M=B*S`.
Only the leading activation dimensions can be dynamic.

All tensors use contiguous `LINEAR` format:

| Index | Tensor | TensorRT type | Shape |
|---|---|---|---|
| 0 | Activation | BF16 | `[M,K]` or `[B,S,K]` |
| 1 | Packed IQ2_XXS payload | INT8 (opaque bytes) | `[N/P,K/256,P,64]` |
| 2 | Base scales | HALF | `[K/256,N]` |
| 3 | Magnitude lookup | BF16 | `[256,8]` |
| Output | Result | BF16 | `[M,N]` or `[B,S,N]` |

`P=128` when `N%128=0`; otherwise `P=8`. Use the
[packing utilities](../../../kernelSrcs/w2a16_gemm_cutedsl/prepare.py)
to prepare weights once. Treat the payload as opaque INT8 storage: do not
quantize, cast numerically, or attach Q/DQ scales to it. Base bytes are viewed
as HALF, and the lookup table is viewed as BF16. Weight tensors can be network
constants, which TensorRT serializes with the engine.

The plugin validates profile bounds, preserves the leading activation shape,
and requests enough TensorRT workspace for intermediate split-K shapes.
Workspace is not monotonic in M. Each context uses TensorRT-owned scratch.
`enqueue` loads the selected AOT module on first use, using the stream passed
by TensorRT to check capture state. After changing an input shape, execute one
uncaptured inference before CUDA Graph capture. Subsequent uses of a loaded
module require no allocation, module loading, or synchronization.

## Run the native tests

Configure with `BUILD_UNIT_TESTS=ON` and the
[runner's AOT build options](../../../kernelSrcs/w2a16_gemm_cutedsl/README.md#build),
provision the [reference fixtures](../../../kernelSrcs/w2a16_gemm_cutedsl/README.md#qualification-status-and-fixtures),
then build and run the plugin tests:

```bash
cmake --build "$BUILD_DIR" --target NvInfer_edgellm_plugin \
  unitTestW2A16GemmPlugin -j 8
export LD_LIBRARY_PATH="$TRT_PACKAGE_DIR/lib:$BUILD_DIR/lib:${LD_LIBRARY_PATH:-}"
EDGELLM_W2A16_GEMM_FIXTURES="$FIXTURE_ROOT" \
  "$BUILD_DIR/bin/unitTests/unitTestW2A16GemmPlugin"
```

The plugin tests validate attribute parsing, tensor formats, rank-2 and rank-3
profile bounds, and workspace requirements. Single-layer engine tests save and
reload serialized constants, change dynamic shapes (including split-K), and
check inference and changed-input CUDA Graph replay against the references. The
[kernel tests](../../../kernelSrcs/w2a16_gemm_cutedsl/README.md#qualification-status-and-fixtures)
validate numerical output and changed-input CUDA Graph replay against independent
checkpoint references. These tests do not establish full-model accuracy or throughput.
