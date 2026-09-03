# CuTe DSL INT4 (W4A16) FP16 GEMM Kernel (Ampere)

This kernel implements the following W4A16 contract:

`C[M, N] = A[M, K] @ dequant(QW[ceil(N/2), K], scales[ceil(K/G), N])^T`

with:

- `A`: FP16 row-major activations `[M, K]`
- `QW`: INT4 weights, repacked offline into a fragment-order `uint32` buffer
- `scales`: FP16 row-major `[ceil(K/G), N]`, group size `G ∈ {32, 128}`
- `C`: FP16 row-major output `[M, N]`
- FP32 accumulation

**Constraints**: `K % 64 == 0`, positive `N`, and `group_size` a multiple of 16
where one of `group_size` / `bK` (=64) divides the other (i.e. `group_size`
divides 64, or is a multiple of 64) — validated for **128 and 32**. The
scale granularity is decoupled from the K-tile: a `bK=64` kernel stages
`groups_per_tile = max(1, bK/group_size)` scale rows per tile (1 for `G≥64`,
2 for `G=32`), so one kernel body serves both group sizes (`group_size` is still
baked at compile time — see below).

Kernel artifacts (static library + headers) are generated locally by
`kernelSrcs/build_cutedsl.py`; CMake links them directly (see
`kernelSrcs/README.md` and `cmake/CuteDsl.cmake`).

## Files

| File | Role |
|---|---|
| `int4_fp16_gemm_ampere.py` | The CuTe DSL kernel (`Int4Fp16GemmAmpere`) + CuPy standalone-test / AOT-export harness. |
| `int4_reference.py` | Torch host-side INT4 unpack/pack, reference matmul, and offline B repack. Used only by the correctness check (the `--export_only` path needs no Torch). |
| `common.py` | Shared CuPy interop, dynamic-tensor marking, and `export_to_c` plumbing. |

## Exported ABI

Each compiled variant exports one C function with the signature:

```
(mA, mQW, mScales, mC, mWorkspace, mLocks, swizzle, stream)
```

- `mQW` — the offline-repacked `uint32` weight buffer
  `[num_n_blocks * num_k_tiles * kn, 128]` (build it with
  `int4_reference.repack_b_for_tile`; the nibble swizzle is done once on the
  host so the in-kernel B staging is a single coalesced `cp.async`).
- `mWorkspace` — placeholder; pass `mC`. The split-K reduction folds into
  `mC` in place (there is no FP32 split-K workspace).
- `mLocks` — zero-initialized `int32` semaphore, `ceil(M/bM) * ceil(N/bN)`
  entries (a 1-element dummy is fine for `split_k == 1`). The last split-K slice
  resets each lock to 0, so the buffer is reusable across launches with no
  per-call memset.
- `swizzle` — runtime `Int32` grouped-M rasterization width (`1` = none).

## Baked configuration space (16 GEMM variants)

An AOT artifact cannot pick a config at runtime, so one exported function is
baked per selected tile / stage / split-K config and the plugin selects per
shape:

| Dimension | Values | Baked? |
|---|---|---|
| CTA tile `(M,N,K)` | `16x128x64`, `32x128x64`, `64x128x64`, `128x128x64` | yes |
| `num_stages` | `2`, `3`, `4` | yes |
| `split_k` | selected values from `1`, `2`, `4` (in-kernel reduction for `>1`) | yes |
| `group_size` | `128` only (kernel also supports `32`) | **yes — but fixed at 128 today** |
| `swizzle` | grouped-M width | **no — runtime arg** |
| `num_k_tiles` | `ceil(K/64)` | **no — dynamic in K** |

The 16-entry set is listed in `_INT4_FP16_GEMM_CONFIGS` in
`kernelSrcs/build_cutedsl.py` and mirrored by the plugin and C++ test X-macros.
All use the single baked `group_size = 128` and names of the form
`int4_fp16_gemm_<tile>_s<stages>_sk<splitk>`. Building another group size, or
shipping both, would multiply the artifact count.

### Important: split-K correctness contract

A baked `split_k = N` is correct **only when `N` divides `ceil(K / 64)`**
(`split_k = 1` always works) — the kernel partitions the K-tiles into `N` equal
static blocks, and a non-dividing factor would silently drop the tail tiles. The
consumer must pick a factor that divides `ceil(K/64)` for the runtime `K`. For
`split_k > 1` the K-slices are reduced **in-kernel** under the `mLocks`
semaphore — no separate reduction kernel and no extra workspace (the running sum
folds into `C`).

## Artifact Development

If you modify this kernel or its registry entries, manually regenerate the
`int4_fp16_gemm` group before running CMake. Otherwise, CMake uses the matching
prebuilt tarball by default. Follow the shared
[CuTe DSL kernel development workflow](../README.md#cute-dsl-kernel-development-workflow)
for the supported Docker and local-venv commands, dependency versions,
cross-compilation, artifact layout, and CMake configuration. This group emits
the 16 selected GEMM variants plus eight small-M GEMV variants.

Supported SMs: **SM80 and newer**. The kernel uses the SM80 instruction floor
(`cp.async` + `mma.sync` 16×8×16 + `ldmatrix`), which is forward-compatible and
runs on Ampere, Ada, Hopper, and Blackwell. `build_cutedsl.py` registers it for
`80 / 86 / 87 / 89 / 100 / 101 / 110 / 120 / 121`. All selected configs fit the
SM86/89 99 KB opt-in shared-memory floor (the largest, `128x128x64` at 4 stages,
is ~81 KB).

> This is the Ampere-ISA path: it runs on Hopper/Blackwell via forward
> compatibility but does **not** use their native tensor-core instructions
> (Hopper `wgmma`, Blackwell `tcgen05`/UMMA), so it is not the throughput-optimal
> kernel on those archs.

## Standalone testing

```bash
cd kernelSrcs/int4_fp16_gemm_cutedsl

# Correctness vs the Torch reference (default tile 16x128x64, no split-K)
python int4_fp16_gemm_ampere.py --mnk 256,512,1024

# A split-K factor (K=1024 -> ceil(K/64)=16, so 2/4/8/16 all divide it)
python int4_fp16_gemm_ampere.py --mnk 256,512,1024 --split_k 4

# A larger tile at 3 stages with grouped-M swizzle
python int4_fp16_gemm_ampere.py --mnk 1024,2048,4096 \
    --cta_tiler_mnk 64,128,64 --num_stages 3 --swizzle 4

# Single-variant AOT export (what build_cutedsl.py invokes per variant)
python int4_fp16_gemm_ampere.py --mnk 256,512,1024 \
    --cta_tiler_mnk 16,128,64 --num_stages 2 --split_k 4 \
    --export_only --output_dir ./out \
    --file_name int4_fp16_gemm_16x128x64_s2_sk4 \
    --function_prefix int4_fp16_gemm_16x128x64_s2_sk4
```

The `--export_only` path is CuPy-only (no Torch); the reference check additionally
imports Torch via `int4_reference`.

## C++ kernel-level accuracy test

| File | Role |
|---|---|
| `unittests/kernelSrcs/int4_fp16_gemm_cutedsl/int4Fp16GemmCuteDslTests.cu` | gtest: random inputs → CPU FP32 reference → launch → compare. Covers all 16 baked GEMM variants plus residual-M and non-64-aligned-N cases; tolerance `relErr < 0.05`. |
| `unittests/kernelSrcs/int4_fp16_gemm_cutedsl/int4Fp16GemvCuteDslTests.cu` | gtest for the eight decode GEMV variants, including non-64-aligned `N`. |

The tests drive the AOT artifact directly, independently of
`Int4GroupwiseGemmPluginV2`: an X-macro table generates module declarations,
one-time loading, and runtime dispatch; the test marshals generated tensor
structs, launches the generated wrapper, and compares against an FP32 reference.
This isolates kernel geometry from plugin tactic selection.

The test is gated on `CUTE_DSL_INT4_FP16_GEMM_ENABLED` and auto-discovered by
CMake (the per-directory glob in `unittests/CMakeLists.txt`), so **no CMake
edits are required** as long as the file lands in a directory that already has a
test group — the group define is emitted generically from `metadata.json`, and
`cute_dsl_setup()` wires the artifact include dir + define onto every test
target.

### Build + run the test

```bash
# After manually generating the full group with the shared workflow:
cmake -B build -DBUILD_UNIT_TESTS=ON -DENABLE_CUTE_DSL=int4_fp16_gemm \
      -DTRT_PACKAGE_DIR=/path/to/TensorRT ..
cmake --build build -j

# Run just the int4 accuracy cases
./build/unittests/unitTestCuteDslKernels --gtest_filter='Int4Fp16GemmAllVariants/*'
```

The tests load and exercise every baked GEMM/GEMV module, so they expect a full
`--kernels int4_fp16_gemm` build.

> **CUDA < 12.8 hosts (x86):** the AOT-generated headers use `cudaLibrary_t`,
> which only exists in the CUDA 12.8+ runtime. As with the other cuteDSL runners,
> add the compat shim define at configure time (the CMake auto-define only covers
> `EMBEDDED_TARGET=jetson-orin`):
> `-DCMAKE_CXX_FLAGS=-DTRT_EDGELLM_CUDA_LIBRARY_T_COMPAT -DCMAKE_CUDA_FLAGS=-DTRT_EDGELLM_CUDA_LIBRARY_T_COMPAT`.
> CI x86 uses CUDA 13.2 so it is not needed there.
