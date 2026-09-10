# Cross-Platform FP16 MoE CuTe DSL Kernels

The `f16_moe` AOT group provides raw FP16 grouped GEMM for
`Fp16MoePlugin`. One device descriptor ABI and one weight layout are shared by
three target-specific implementations:

| Variant | Target SMs | Mainloop |
|---|---|---|
| `f16_moe_ampere_grouped_fp16` | 80, 86, 87, 89 | `cp.async`, LdMatrix, warp MMA |
| `f16_moe_blackwell_grouped_fp16` | 100, 101, 103, 110 | TMA, tcgen05, TMEM |
| `f16_moe_blackwell_geforce_grouped_fp16` | 120, 121 | TMA, LdMatrix, warp MMA |

Every variant computes `D[M,N] = A[M,K] * B[N,K]^T` with row-major FP16
operands/output and FP32 accumulation. The same module is invoked for FC1 and
FC2. Activation and route reduction are intentionally separate CUDA kernels.
The Blackwell datacenter implementation is adapted from the CUTLASS 4.4.2
CuTe DSL grouped-GEMM example and retains its BSD-3-Clause notice; the repository
exports all three variants with its existing `nvidia-cutlass-dsl==4.7.0`
toolchain rather than adding a second C++ GEMM backend.

## Device descriptor ABI

All arrays remain in GPU memory:

- `problem_shapes`: INT32 `[E,4]`, containing `(M,N,K,L)` with `L=1`.
- `strides`: INT32 `[E,3,2]`, containing row/contiguous strides for A/B/D.
- `addresses`: INT64 `[E,3]`, containing A/B/D device addresses.
- `scratch`: UINT8 `[E,3,128]`; TMA variants use one A/B/D descriptor set
  per persistent CTA and Ampere keeps the argument for ABI parity.

`E` is the runtime expert count (`group_count`); the AOT trace sizes its dummy
buffers with `MAX_NUM_EXPERTS` (256), which bounds `group_count` but is not
baked into the cubin.

After the seven device pointers, the exported wrapper accepts these runtime
values in order:

- `max_m`, `max_n`, and `max_k`: global bounds used to construct the A, B, and
  D tensor views. They do not replace the exact per-expert `(M,N,K)` values in
  `problem_shapes`.
- `group_count`: the number of descriptor groups (runtime expert count). The
  cubin is runtime-polymorphic in this dimension; the C++ runner restricts it
  to the product-supported set `{128, 256}`
  (`CuteDslF16MoeRunner::kSupportedNumExperts`).
- `max_active_clusters`: the cached persistent launch count for the device.
- CUDA stream.

The exact problem shapes and A/B/D addresses remain device resident, so the
persistent scheduler can skip empty experts without host synchronization.

## Artifact Development

If you modify this kernel or its registry entries, manually regenerate the
`f16_moe` group before running CMake. Otherwise, CMake uses the matching
prebuilt tarball by default. Follow the shared
[CuTe DSL kernel development workflow](../README.md#cute-dsl-kernel-development-workflow)
for the supported Docker and local-venv commands, dependency versions,
cross-compilation, artifact layout, and CMake configuration.

The selected target SM determines which implementation family is exported.
The target CPU architecture can differ from the build host; the shared build
wrapper performs the corresponding host-object cross-compilation.

## Fixed plugin contract

- FP16 only, expert count in `{128, 256}`, top-K 1 through 8.
- `H % 128 == 0`, `I % 64 == 0`, and `FC1_N % 128 == 0`.
- FC1 weights: row-major `[E,FC1_N,H]`.
- FC2 weights: row-major `[E,H,I]`.
- SwiGLU FC1 rows: `[up0:64,gate0:64,up64:128,gate64:128,...]`.
- Route index: `expanded_row = token * top_k + slot`.

BF16, biases, Hopper SM90, quantization scales, expert parallelism, and tensor
parallelism are not part of this group.
