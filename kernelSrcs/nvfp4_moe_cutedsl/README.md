# SM110 NVFP4 MoE CuTe DSL Kernels

This directory contains the Thor SM110 split FC1 / FC2 CuTe DSL backend used by
the unified [`Nvfp4MoePlugin`](../../cpp/plugins/nvfp4MoePlugin/). It keeps the
plugin tensor ABI and exports the `nvfp4_moe` artifact group from
[`kernelSrcs/build_cutedsl.py`](../build_cutedsl.py); today the only variants
in that group target SM110/Thor. The SM120 / SM121 fused decode + prefill
kernels live in
[`kernelSrcs/nvfp4_fused_moe_cutedsl/`](../nvfp4_fused_moe_cutedsl/) and feed
the same plugin via the `nvfp4_fused_moe` group.

Pipeline produced by the AOT pack:

1. **FC1 (gather grouped GEMM + activation + FP4 requant)** — gathers routed
   tokens by expert, runs the FP4 blockscaled grouped GEMM, applies SwiGLU /
   ReLU², and requantizes the intermediate activation back to NVFP4 (packed
   FP4 bytes + FP8-E4M3 scale-factor blocks).
2. **FC2 (grouped GEMM + router-scale finalize / scatter)** — applies
   `alpha = input_gsf * weight_gsf` in the kernel epilogue, multiplies by the
   per-token router weight, and scatter-reduces the result back to the original
   token layout, emitting `[T, H]` FP16.

The legacy `nvfp4_moe` prefill/decode N-major sources and the legacy
`Nvfp4MoePlugin` Marlin path are no longer part of this backend.

## Supported Hardware

| GPU | SM | Status |
|---|---|---|
| NVIDIA Thor | SM110 | Primary target (aarch64 cross build) |

The kernels can also be exported on a Blackwell datacenter card (SM100 / SM103)
for local iteration, but the only AOT artifact shipped is `aarch64/sm_110/`.
Blackwell datacenter prefill / decode lives in the `nvfp4_fused_moe` group; the
SM120 / SM121 GeForce path lives in `nvfp4_fused_moe_cutedsl/`.

## Kernel Variants

The `nvfp4_moe` group exports 6 AOT-compiled kernel objects (`.o` + `.h`
pairs):

### FC1 — Gather grouped GEMM + activation + FP4 requant (4 variants)

| Variant | Activation | MMA N-tile |
|---|---|---|
| `nvfp4_moe_sm110_fc1_relu2_n128`  | ReLU²  | 128 |
| `nvfp4_moe_sm110_fc1_relu2_n256`  | ReLU²  | 256 |
| `nvfp4_moe_sm110_fc1_swiglu_n128` | SwiGLU | 128 |
| `nvfp4_moe_sm110_fc1_swiglu_n256` | SwiGLU | 256 |

### FC2 — Finalize with scatter-reduce (2 variants)

| Variant | MMA N-tile |
|---|---|
| `nvfp4_moe_sm110_fc2_n128_fp16` | 128 |
| `nvfp4_moe_sm110_fc2_n256_fp16` | 256 |

FC2 outputs FP16 directly back into the token layout. The runner currently
picks `n128` unconditionally (see `selectMmaTilerN` in
[`cuteDslNvfp4MoeSm110Runner.cpp`](../../cpp/kernels/moe/nvfp4_cutedsl/cuteDslNvfp4MoeSm110Runner.cpp));
`n256` is exported for follow-up benchmarking.

## Tensor Contract

The AOT pack is specialized for the current Thor Qwen3 / Nemotron contract:
`num_experts = 128` and `top_k = 8`. Hidden size and intermediate size remain
runtime dimensions.

- FC1 weights: `[E, N1, H / 2]` (N1 = `2 * I` for SwiGLU, `I` for ReLU²)
- FC1 scales:  `[E, ceil(N1 / 128), ceil((H / 16) / 4), 32, 4, 4]`
- FC2 weights: `[E, H, I / 2]`
- FC2 scales:  `[E, ceil(H / 128), ceil((I / 16) / 4), 32, 4, 4]`
- Grouped MoE metadata: tile-to-expert, tile limits, permuted-to-expanded row mapping
- Output: FP16 `[T, H]`

Alignment rules enforced by the SM110 runner: `H % 128 == 0`, `I % 64 == 0`,
`FC1_N % 128 == 0`.

## Artifact Development

If you modify this kernel or its registry entries, manually regenerate the
`nvfp4_moe` group before running CMake. Otherwise, CMake uses the matching
prebuilt tarball by default. Follow the shared
[CuTe DSL kernel development workflow](../README.md#cute-dsl-kernel-development-workflow)
for the supported Docker and local-venv commands, dependency versions,
cross-compilation, artifact layout, and CMake configuration.

CMake auto-defines `CUTE_DSL_NVFP4_MOE_ENABLED`; the plugin uses this backend
whenever `getSMVersion() == 110`.

## Standalone Kernel Compilation

To compile a single variant manually (debugging / iteration):

```bash
cd kernelSrcs

# FC1: ReLU², N-tile = 128
python nvfp4_moe_cutedsl/export_fc1_kernel.py \
  --activation relu2 --mma_tiler_n 128 \
  --dummy-experts 128 --dummy-top-k 8 \
  --output_dir ./out \
  --file_name nvfp4_moe_sm110_fc1_relu2_n128 \
  --function_prefix nvfp4_moe_sm110_fc1_relu2_n128 \
  --export_only

# FC1: SwiGLU, N-tile = 128
python nvfp4_moe_cutedsl/export_fc1_kernel.py \
  --activation swiglu --mma_tiler_n 128 \
  --dummy-experts 128 --dummy-top-k 8 \
  --output_dir ./out \
  --file_name nvfp4_moe_sm110_fc1_swiglu_n128 \
  --function_prefix nvfp4_moe_sm110_fc1_swiglu_n128 \
  --export_only

# FC2: N-tile = 128, FP16 output
python nvfp4_moe_cutedsl/export_fc2_kernel.py \
  --mma_tiler_n 128 --output_dtype fp16 \
  --dummy-experts 128 --dummy-top-k 8 \
  --output_dir ./out \
  --file_name nvfp4_moe_sm110_fc2_n128_fp16 \
  --function_prefix nvfp4_moe_sm110_fc2_n128_fp16 \
  --export_only
```

`--activation` accepts `relu2` and `swiglu`; `--mma_tiler_n` accepts `128` and
`256`; FC2's `--output_dtype` accepts `bf16` and `fp16` (the AOT pack only
ships `fp16`).

## Python Kernel-Class Usage

The two AOT export scripts in
[Standalone Kernel Compilation](#standalone-kernel-compilation) wrap the
underlying CuTe DSL kernel classes. When iterating on a single variant (e.g. adding a new activation,
debugging a TMEM layout), it is often easier to instantiate the class directly
in Python instead of going through `export_fc{1,2}_kernel.py`. The snippets
below match the `Example:` blocks in the kernel module docstrings.

### FC1 — gather grouped GEMM + activation + FP4 requant

`BlockScaledContiguousGatherGroupedGemmKernel` lives in
[`blockscaled_contiguous_gather_grouped_gemm_act_fusion.py`](blockscaled_contiguous_gather_grouped_gemm_act_fusion.py).
`use_2cta_instrs` is inferred from `mma_tiler_mn[0]` (`True` when M=256, `False`
when M=128):

```python
from blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
    BlockScaledContiguousGatherGroupedGemmKernel,
)

gemm = BlockScaledContiguousGatherGroupedGemmKernel(
    sf_vec_size=16,
    mma_tiler_mn=(256, 128),  # use_2cta_instrs=True since M=256
    cluster_shape_mn=(2, 1),
    vectorized_f32=True,
)
gemm(
    a=a_tensor,
    b=b_tensor,
    c=c_tensor,
    sfa=sfa_tensor,
    sfb=sfb_tensor,
    sfc_tensor=None,
    input_global_scale_tensor=input_global_scale_tensor,
    down_input_scale_tensor=None,
    tile_idx_to_expert_idx=tile_idx_to_expert_idx,
    tile_idx_to_mn_limit=tile_idx_to_mn_limit,
    token_id_mapping_tensor=token_id_mapping_tensor,
    num_non_exiting_tiles=num_non_exiting_tiles,
    alpha=alpha,
    max_active_clusters=max_active_clusters,
    stream=stream,
)
```

### FC2 — grouped GEMM + router-scale finalize / scatter

`Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel` lives in
[`blockscaled_contiguous_grouped_gemm_finalize_fusion.py`](blockscaled_contiguous_grouped_gemm_finalize_fusion.py):

```python
from blockscaled_contiguous_grouped_gemm_finalize_fusion import (
    Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
)

gemm = Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel(
    sf_vec_size=16, mma_tiler_mn=(256, 128), cluster_shape_mn=(2, 1),
)
gemm(
    a_tensor, b_tensor, sfa_tensor, sfb_tensor, out_tensor,
    max_active_clusters, stream,
)
```

Both classes enforce the constraints listed in their docstrings — MMA tiler M
∈ {128, 256}, MMA tiler N ∈ {64, 128, 192, 256}, cluster shape M/N positive
powers of two with total cluster size ≤ 16 (and ≤ 4 for scale-factor
multicasts). The SM110 AOT pack only ships `m_tile_size = 128` (1-CTA, see
[Important Notes](#important-notes)); FC1 and FC2 vary along the N tile via
`mma_tiler_mn = (128, 128)` and `(128, 256)` (see
[Kernel Variants](#kernel-variants)). The docstring examples above show
`(256, 128)` purely to illustrate the 2-CTA `use_2cta_instrs` path and are
**not** the configuration shipped by [`build_cutedsl.py`](../build_cutedsl.py)
for SM110.

## Important Notes

- **Alpha scaling.** FC1 applies per-expert `alpha = input_gsf * weight_gsf`
  inside the kernel epilogue, before the fused activation. Alpha is a `[E]`
  FP32 tensor on the plugin's input slot.
- **SwiGLU weight interleave.** SwiGLU FC1 weights must be laid out as 64-row
  interleaved `(up, gate)` chunks along the N axis (`moe_inter_size = 2 * I`).
  Plain `[up..., gate...]` concatenation produces wrong results silently.
  See `repack_nvfp4_gated_moe_experts` in
  [`tensorrt_edgellm/checkpoint/repacking.py`](../../tensorrt_edgellm/checkpoint/repacking.py).
- **PDL.** Programmatic Dependent Launch is currently disabled
  (`EDGELLM_ENABLE_PDL = False` in [`cute_utils.py`](cute_utils.py)).
- **Tile size.** Only `m_tile_size = 128` (1-CTA) is supported.

## Validation

The SM110 NVFP4 MoE contract is validated by the
[`CuteDslNvfp4MoeSm110Test`](../../unittests/cpp/kernels/moe/nvfp4MoeCuteDslSm110Tests.cu)
smoke and accuracy tests.

## File Map

| File | Description |
|---|---|
| `blockscaled_contiguous_gather_grouped_gemm_act_fusion.py` | FC1 kernel: gather grouped GEMM + activation + FP4 requant |
| `blockscaled_contiguous_grouped_gemm_finalize_fusion.py`   | FC2 kernel: grouped GEMM + router-scale finalize / scatter |
| `export_fc1_kernel.py` | FC1 AOT export script (invoked by `build_cutedsl.py`) |
| `export_fc2_kernel.py` | FC2 AOT export script (invoked by `build_cutedsl.py`) |
| `export_common.py`     | Shared AOT export helpers (dummy pointers, SF buffer sizing) |
| `custom_pipeline.py`   | SM110 CuTe DSL pipeline helper |
| `cute_utils.py`        | CuTe DSL utility helpers (PTX helpers, PDL gate, etc.) |
| `moe_compat.py`        | Compatibility helpers for the split SM110 path |
