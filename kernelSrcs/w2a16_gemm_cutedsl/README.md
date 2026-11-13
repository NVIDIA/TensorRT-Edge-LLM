<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Blackwell GeForce W2A16 dense GEMM

This kernel/runner supports SM120 GPUs, GB10 (SM121, 48 SMs) and Thor (SM110). It computes
`Y[M,N] = A[M,K] @ dequant(W[N,K]).T` with BF16 activations/output and FP32
accumulation. Outside the small-M path below, reconstructed weights round once to BF16
before contraction.

The compute path comes from b12x commit
[`2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88`](https://github.com/local-inference-lab/b12x/tree/2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88).
The standalone source requires CuTeDSL 4.8.0 and CUDA 13; it does not import b12x.
The BF16 Tensor Core engine retains its BSD-3-Clause license. The other ported
helpers retain Apache-2.0 attribution, and the magnitude table carries the GGML
MIT notice below.

## Prepare weights

[prepare.py](prepare.py) accepts contiguous uint8 raw blocks `[N,K/256,66]`
on CPU or CUDA. It returns owned, compact storage:

- Payload bytes `[N/B,K/256,B,64]`, where `B=128` when `N%128=0`, otherwise `B=8`.
- FP16 base bytes `[K/256,N,2]`, without padding.
- `make_lookup_table(device)` separately creates a 4096-byte BF16 magnitude table.

Packed weights retain exactly 66 bytes per 256 weights. Packing validates finite
FP16 bases and must finish before inference/capture. Payload/base planes and the
lookup table must remain alive and unchanged while execution uses them.

## Dispatch and runner contract

The shape contract is `1 <= M <= 32768`, positive `K%256=0`, and positive
`N%8=0`. The runner also limits `N <= 65535*64` to respect CUDA's grid.z limit.
M, N, and K are runtime arguments. The AOT variants specialize packing,
tile geometry, and split-K count, without baking model dimensions into objects.

- M<=8 runs the CUDA kernel in
  [w2a16SmallMGemm.cu](../../cpp/kernels/gemm/w2a16_cutedsl/w2a16SmallMGemm.cu)
  on the same packed weights and without workspace. Each 32-weight sub-block
  is contracted on unscaled signed magnitudes with BF16 `mma.sync`, and its
  FP32 partial is then multiplied by the sub-block scale, so this path does not
  round reconstructed weights to BF16.
- The AOT variants use the pinned upstream IQ2_XXS policy tuned for GB10 (48 SMs);
  SM120 and SM110 use the same policy. Small M uses SIMT GEMV when N uses
  eight-row packing, or when M<=2 and N>=K or K<=1024; remaining M<=8 with
  N>=6144 uses the transposed M8/N64/K256 Tensor Core tile; other cases use
  M16/N64/K128 for M<=16 and M16/N128/K256 for M>16. Because the CUDA kernel
  serves M<=8, the runner dispatches only the M>8 part of this policy.
- Split-K selects 1, 2, 4, or 8 partitions using upstream's grid-wave cost and
  minimum K-per-partition of 512. FP32 partials reduce to BF16 on the same stream.

Use [CuteDslW2A16GemmRunner](../../cpp/kernels/gemm/w2a16_cutedsl/cuteDslW2A16GemmRunner.h):

1. Check `canImplement` for the shape and device.
2. Allocate the caller-owned output and `getWorkspaceSize` bytes of scratch.
   Zero workspace is valid for an unsplit call. Scratch must be 256-byte aligned.
3. Call `ensureKernelModules(params, stream)` before capture for every selected
   variant. Only dimensions are needed; preload never accesses tensor pointers.
4. Call `run` with an explicit non-default stream. An unprepared variant returns
   `cudaErrorInitializationError`; `run` never loads modules.

Input pointers must be contiguous, 16-byte aligned, and resident on the current
device. Device ownership and allocation extents are caller preconditions. The
runner rejects null/misaligned pointers, undersized scratch, and output/scratch
ranges overlapping inputs or each other. Concurrent calls need separate outputs
and scratch; calls on one stream can reuse scratch without clearing it.

## Build

From the repository root, use the existing [CuTeDSL build flow](../README.md):

```bash
python kernelSrcs/build_cutedsl.py --gpu_arch sm_121 --arch aarch64 \
  --kernels w2a16_gemm --cuda-version 13 --output_dir "$ARTIFACT_DIR"
```

The metadata-driven CMake integration supplies `CUTE_DSL_W2A16_GEMM_ENABLED`
and `cutedsl_w2a16_gemm_all.h`. Place the generated target artifact under the
normal `cpp/kernels/cuteDSLArtifact/aarch64/sm_121` location and select
`-DENABLE_CUTE_DSL=w2a16_gemm -DEMBEDDED_TARGET=gb10` when configuring CMake.

For an x86-64 SM120 GPU, generate the matching artifact instead:

```bash
python kernelSrcs/build_cutedsl.py --gpu_arch sm_120 --arch x86_64 \
  --kernels w2a16_gemm --cuda-version 13 --output_dir "$ARTIFACT_DIR"
```

Use `cpp/kernels/cuteDSLArtifact/x86_64/sm_120` and configure with
`-DENABLE_CUTE_DSL=w2a16_gemm`, `-DCUTE_DSL_ARTIFACT_TAG=sm_120`, and
`-DCMAKE_CUDA_ARCHITECTURES=120a`. For Thor, generate `--gpu_arch sm_110 --arch aarch64`
and configure with `-DEMBEDDED_TARGET=jetson-thor`; the kernels use the same
warp-level BF16 `mma.sync` and TMA path. Variant names include `blackwell_geforce`
to distinguish this backend from future implementations. The Tensor Core path
uses TMA, so this backend does not support SM80/86/87/89.


## Qualification status and fixtures

The native `unitTestKernelsW2A16Gemm` target checks 162 shape cases against
independently decoded checkpoint references, including changed-input CUDA Graph
replay and poisoned workspace reuse. Two additional tests reject invalid shapes
and host contracts. The numerical gates are relative L2 <= 0.01 and cosine >=
0.999. Provision fixtures before running; missing fixture configuration skips
numerical tests, and missing or incompatible files fail them.

The [TensorRT plugin](../../cpp/plugins/w2a16GemmPlugin/README.md) supports
serialized single-layer engines. Model
export and full-model qualification remain follow-up work. Kernel and plugin
tests do not establish model accuracy or throughput.

The [fixture generator](../../unittests/resources/gen_w2a16_gemm_fixtures.py)
requires explicit projection keys and an immutable checkpoint revision:

For the native suite, select two dense projection tensors with N/K equal to
4096/6144 and 6144/4096, in that order, from the same checkpoint shard. Set
`DOWN_PROJECTION_KEY` and `UP_PROJECTION_KEY` to those exact tensor keys, and
create a new `FIXTURE_ROOT` outside the source tree:

```bash
python unittests/resources/gen_w2a16_gemm_fixtures.py \
  --checkpoint-file "$CHECKPOINT_SHARD" --revision "$CHECKPOINT_REVISION" \
  --tensor "$DOWN_PROJECTION_KEY" --tensor "$UP_PROJECTION_KEY" \
  --rows 1 2 3 4 8 9 16 17 128 512 2048 4096 6144 8192 10240 12288 \
    14336 16384 18432 20480 22528 24576 26624 28672 30720 32768 \
  --output "$FIXTURE_ROOT/fixtures-primary"

python unittests/resources/gen_w2a16_gemm_fixtures.py \
  --checkpoint-file "$CHECKPOINT_SHARD" --revision "$CHECKPOINT_REVISION" \
  --tensor "$DOWN_PROJECTION_KEY" \
  --slice 8:256 --slice 136:768 --slice 128:4096 --slice 256:6144 \
  --slice 1024:6144 --slice 136:4096 --slice 520:6144 --slice 1048:6144 \
  --slice 512:6144 --slice 1536:6144 --slice 1480:6144 \
  --rows 1 2 3 4 8 9 16 17 128 8192 \
  --output "$FIXTURE_ROOT/fixtures-boundary-full"

# Configure with BUILD_UNIT_TESTS=ON and the CuTeDSL options above.
cmake --build "$BUILD_DIR" --target unitTestKernelsW2A16Gemm -j 8
EDGELLM_W2A16_GEMM_FIXTURES="$FIXTURE_ROOT" \
  "$BUILD_DIR/bin/unitTests/unitTestKernelsW2A16Gemm"
```

The boundary fixtures preserve exact rows and 256-weight blocks from the first
projection. This matrix reaches all 21 default-dispatch variants; the slices
are functional coverage, not additional model projections.

Repeat `--tensor` for multiple projections in the same shard. The manifest
records exact N/K, raw payload SHA-256, and per-file SHA-256. Each file starts
with four little-endian uint32 values: magic `0x57324431`, N, K, and M.
`weights.bin` uses M=0 and stores payload, bases, and the magnitude table.
Each `m<M>.bin` stores `input0, expected0, input1, expected1` in BF16, with an
independent raw-block decoder and FP32 matrix multiplication as the oracle.
The two inputs support changed-input CUDA Graph replay validation. Synthetic
codec fixtures can exercise tooling but cannot replace real checkpoint gates.

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
