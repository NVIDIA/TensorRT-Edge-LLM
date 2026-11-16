<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DIGITS W2A16 MoE kernels

This directory contains the IQ2_XXS/BF16 compute implementation ported from b12x:
compressed-weight decode, BF16 Tensor Core GEMMs, FC1 ReLU², split-K reduction,
the resident-grid barrier, and route reduction. Build and runtime do not import b12x.

## Supported contract

- NVIDIA Thor (SM101/SM110) and fused AOT target SM121; CuTeDSL 4.8.0.
  The runner normalizes Thor's legacy SM101 identifier to SM110. UMMA accepts
  any positive SM count on Thor. Fused AOT uses a fixed 48-SM grid configuration
  and requires at least 48 SMs on the target device.
  Device measurements use CUDA 13 and SM110 artifacts.
- H=1024, topK=9, ReLU², BF16 input/output. No expert parallelism.
- M=1..8192 for both Thor UMMA and fused AOT. Thor uses UMMA for standard
  top-k routing across all E/I pairs below.
- Fused AOT M=1..8: direct routing, fused FC1/FC2, in-kernel FP32 top-k sum.
- Fused AOT prefill, M>=9 up to the device limit: existing Edge-LLM route packing
  (8/16/32/48/64 slots per block), fused FC1/FC2,
  and existing BF16 slot aggregation.
- Mode and row tile select the AOT variant; M, E and I are runtime arguments.
  The target SM selects a compile-time hardware configuration.
  Prefill selects the first tile whose 90% capacity exceeds M*topK/E,
  falling back to 64, following the b12x shape-based policy.
  Row tiles 8/16 use K=128, N=128 with two pipeline stages;
  row tiles 32/48/64 use K=64, N=256 with four stages for both FC1 and FC2.

| E | I |
|---|---|
| 256 | 1536, 2048, 2304, 2560 |
| 320 | 2560, 2816 |
| 384 | 2304, 2560, 2816 |
| 448 | 2304, 2560, 2816 |
| 512 | 2560, 2816 |

This is a kernel/runner integration. Model export, TensorRT plugin dispatch,
engine integration, and full-model/few-layer qualification are separate work.

## Weight representation and precision

Raw checkpoint tensors are uint8 `[E,N,K/256,66]`: one FP16 base followed by
64 bytes of grid/sign/scale records per 256 weights. `prepare.pack_weights`
losslessly reorders them into int32 payload `[E,N/64,K/128,8,4,8,2]` and FP16
bases `[E,K/256,N/16,8,2]`. It does not requantize. FC1 uses N=I,K=1024;
FC2 uses N=1024,K=I. `prepare.make_lookup_table` creates a 2048-byte selector LUT.
`unitScales` is an FP32 vector of E ones. Prepare these once before inference.

Weights decode to BF16 and GEMMs accumulate in FP32. FC1 rounds to BF16 before
ReLU² and rounds the activation to BF16. FC2 rounds its result and router
multiplier to BF16 and stores BF16 route contributions. Top-k reduction
accumulates those contributions in FP32 and converts once to BF16.

All input pointers must be 16-byte aligned, contiguous, and on the current device.
Expert IDs are int32 `[M,9]` in `[0,E)`; routing weights are finite FP32 `[M,9]`.
The caller supplies a 256-byte-aligned workspace sized by
`CuteDslW2A16MoeRunner::getWorkspaceSize`. Input, output, weights, and workspace must not alias.
Concurrent streams need distinct workspaces. The runner clears barrier state
on each call. Once modules are loaded, enqueue performs no allocations or
host/device synchronization.

## Native runtime flow

`CuteDslW2A16MoeParams` carries runtime dimensions and non-owning device pointers.
The runner follows the same lifecycle as `CuteDslNvfp4MoeRunner`:

1. Query the execution device once with `getSMVersion()` and
   `getDeviceMultiProcessorCount()`. Pass the same `smVersion` and `numSms` to
   `canImplement`, workspace sizing, preload, and execution.
2. Allocate caller-owned scratch using `getWorkspaceSize(M, E, I, smVersion, numSms)`.
3. Call `ensureKernelModules(params, smVersion, numSms, stream)` outside graph capture for each
   mode/row-tile combination that will be used. Only dimensions are required; it neither accesses
   tensor pointers nor launches a kernel.
4. Call `run(params, workspace, workspaceBytes, smVersion, numSms, stream)`, eagerly or in a graph.

`run` validates inputs and ensures the selected module is loaded before touching
workspace/output. Thor selects the native SM110a UMMA CuTe DSL implementation
for every supported E/I pair. Its inputs must use standard top-k routing:
each token's nine expert IDs must be distinct on every graph replay. The runner
does not inspect device IDs on the CPU.
Other targets use fused AOT `runDecode` for M<=8 or `runPrefill` for M>8.

The UMMA path packs routes, runs the FC1 and FC2 projections, and reduces nine
route contributions. Every M uses a single-stage 48-token tile, a
96-token tile once experts average more than 48 routed rows, or a 192-token tile
once they average more than 96.
All keep the decoded weight operand in TMEM.
The 48-token tile runs a persistent grid of at most 200 CTAs. Each CTA claims
(expert, 128-column) work items for experts that have routes through an atomic
counter that the pack kernel resets, so skewed routing does not launch CTAs for
idle experts. For M<=16 each CTA allocates a 32-column TMEM block; an expert
with at most 16 routes uses a 16-column accumulator and sizes its MMA,
activation loads and epilogue to 16 tokens. It issues each K tile's weight
loads seven tiles before decoding it.
Each tile keeps 4 (48-token), 8 (96-token) or 16 (192-token) copies of the
selector LUT in disjoint shared-memory bank sets, and each lane reads one copy,
which reduces bank conflicts on the random grid lookups.
The 96- and 192-token tiles are warp specialized: groups of four warps decode
weights into TMEM stages and run the epilogue, one warp per group loads
activations into shared-memory stages, and one thread issues the MMAs.
Per-stage mbarriers link them, so decoding, activation loads and MMAs of
different K tiles overlap. Each activation stage is published by the
completion of its own `cp.async` copies, so an MMA never waits for later loads
to be issued. The 96-token tile uses two weight and three activation stages;
the 192-token tile uses two decode groups that take alternate K tiles, with four
weight and four activation stages, and still fits two CTAs per SM.
The 192-token tile runs as a two-CTA cluster that issues 256-row
`cta_group::2` MMAs: each CTA decodes its own 128 weight rows and loads half
of the expert's tokens, so each CTA reads half the activations that it would
read without the pair. The even CTA issues the MMAs; the odd CTA forwards its
readiness barriers to it.
The 96- and 192-token tiles also run FC1 and FC2 in a single persistent launch:
20 clusters for the 192-token pair, or 80 CTAs for the 96-token tile. A work
unit is one token chunk of an expert. Each unit contributes one FC1 item per
128 intermediate columns of each CTA, and FC2 items that cover the 1024 output
columns. A unit's FC2 items are ordered after the FC1 items of the unit 4
places later for the 192-token tile, or 12 places later for the 96-token tile,
whose 80 CTAs keep more units in flight. The MMA thread of each single CTA, or
of the even CTA in a pair, claims items in that order from an atomic counter
and broadcasts them to its CTAs. An FC2 item's load warps wait until all FC1
CTAs of its unit have published their columns. FC2 therefore reads intermediate
activations soon after they are written, while they are still in L2. Because
items are claimed in dependency order, every item that an FC2 item waits on
has already been claimed by a running CTA, so the launch never deadlocks.
Its source is `umma_kernel.py`, with PTX primitives in `umma_intrinsics.py`.
The build driver exports its three variants for SM101a or SM110a.
CMake derives `CUTE_DSL_W2A16_MOE_ARTIFACT_SMS` from the artifact's target SMs
(SM101, SM110, SM120 or SM121) and enables the UMMA path
(`CUTE_DSL_W2A16_MOE_THOR_ENABLED`) and the fused AOT path
(`CUTE_DSL_W2A16_MOE_FUSED_ENABLED`) for the SMs that need them. A combined
artifact such as `sm_110_sm_120` carries both and the runner picks one from the
device SM: Thor uses UMMA, SM120 and SM121 use fused AOT. A device whose SM is
not in the artifact is rejected. The runner matches the device after
normalizing Thor's legacy identifier.
Build the artifacts and runner together.
The workspace query sizes the supplied hardware's implementation. Thor reserves
route metadata and intermediate activations; fused AOT also reserves split-K scratch
and barrier state.
The fused AOT grid uses 48 SMs multiplied by the variant's resident blocks per
SM. Scratch sizes and barrier offsets use the same SM count. `getWorkspaceSize` uses the supplied hardware values;
allocate and run on the corresponding device. AOT decode clears barrier state and launches the
fused MLP directly. AOT prefill clears
barrier state, builds the packed routes, launches the fused MLP, and aggregates
the route outputs. Workspace sizing and pointer binding use one byte-offset
layout. The W2A16 kernel, dispatch threshold, and workspace remain independent
of the W4A4 backend.

## Build

Use the CUDA 13 environment from the parent [build instructions](../README.md).
The `w2a16_moe` group exports three UMMA variants (N48, N96 and N192) for
SM101/SM110, or six fused AOT variants (one decode and five prefill) for SM120
and SM121. The runner compiles the paths that the artifact carries:

```bash
python kernelSrcs/build_cutedsl.py --gpu_arch sm_121 --arch aarch64 \
    --kernels w2a16_moe --cuda-version 13 --output_dir "$ARTIFACT_DIR"
```

For Thor with CUDA 13, use `--gpu_arch sm_110` with the same kernel group.
Legacy toolchains that target SM101a use `--gpu_arch sm_101 --cuda-version 12`.
The build driver preserves the toolchain's architecture name when compiling;
runtime normalization does not retarget the compiled binary. Rebuild the AOT
artifacts, native runner, and its callers together.
The launch ABI is unchanged.

Normal CuTeDSL CMake integration derives `CUTE_DSL_W2A16_MOE_ENABLED` and
`cutedsl_w2a16_moe_all.h` from metadata. The runner is in
`cpp/kernels/moe/w2a16_cutedsl/`, guarded by that compile definition.

## Validation

`unittests/cpp/kernels/moe/w2a16MoeCuteDslTests.cpp` directly exercises the runner
in the existing `unitTestKernelsMoe` GoogleTest target. Offline fixtures use
actual Bolt IQ2_XXS checkpoint bytes and an independent GGML decoder with
PyTorch FP32 GEMMs. Admission thresholds are relative L2 <=0.01 and cosine
>=0.999. Cases cover all 14 E/I pairs, balanced/hot/imbalanced routes,
decode/prefill, changed graph inputs/routes, zero routing weights, workspace
reuse, shape-only preload followed by graph capture before an eager launch,
token-count boundaries, and invalid host/hardware contracts.

Generate fixtures once in a CUDA-enabled Python environment with PyTorch and
safetensors. Build the AOT artifacts into `cpp/kernels/cuteDSLArtifact` using
the command above, then build and run the normal CMake test target:

```bash
python unittests/resources/gen_w2a16_moe_fixtures.py \
    --checkpoint "$CHECKPOINT_DIR" --output "$FIXTURE_DIR"
cmake -S . -B build -DBUILD_UNIT_TESTS=ON -DENABLE_CUTE_DSL=w2a16_moe \
    -DEMBEDDED_TARGET=jetson-thor -DCUTE_DSL_ARTIFACT_TAG=sm_110 \
    -DTRT_PACKAGE_DIR="$TRT_PACKAGE_DIR"
cmake --build build --target unitTestKernelsMoe -j
EDGE_LLM_W2A16_TEST_DATA="$FIXTURE_DIR" \
    build/bin/unitTests/unitTestKernelsMoe --gtest_filter='*W2A16*'
```

GPU cases skip on unsupported hardware or when `EDGE_LLM_W2A16_TEST_DATA` is
unset. When it is set, missing or incompatible fixtures fail the tests.

## GGML table license

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
