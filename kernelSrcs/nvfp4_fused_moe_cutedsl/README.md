# NvFP4 Fused MoE CuTe DSL Kernels (SM120/SM121)

End-to-end fused Mixture-of-Experts kernels for Blackwell consumer
GeForce silicon (SM120 / SM121). Each kernel fuses route/pack + FC1 +
activation + quantize + FC2 + scatter into a single resident-grid launch,
eliminating the host-side FC1/FC2 handoff of the SM110 decomposed
`nvfp4_moe` pipeline. Both backends are exposed through
`NvFP4MoEPluginGeforce` (see
[`cpp/plugins/nvfp4MoePluginGeforce/`](../../cpp/plugins/nvfp4MoePluginGeforce/)).

## Shape support

The kernel wrappers are **shape-polymorphic** in `H` (hidden_size),
`I` (moeInterSize), `E` (numExperts), and `top_k`: these are passed as
runtime `Int32` arguments and the TMA descriptors are rebuilt at each
launch. Hidden size must satisfy `H > 0 && H % (tile_k * ab_stage) == 0`
(currently 256, with `tile_k = sf_vec_size * 8 = 128` and the AOT-baked
static `ab_stage = _AB_STAGE_DEFAULT = 2`). The C++ runner's
`CuteDslNvfp4MoeRunner::canImplement` (via `kHiddenSizeAlignment`) is the
single point of enforcement so the K-tile pipeline always drains cleanly.
Validated for H in `{1024, 2048}`; other multiples of 256 are expected to
work by shape polymorphism but should be accuracy-checked.

## Backends

| Backend | Script | Best for |
|---|---|---|
| **Decode** | `export_decode_kernel.py` | Small routed sets (`routed_rows <= 640`). Resident-grid barrier between route/pack and compute phases. |
| **Prefill** | `export_prefill_kernel.py` | Large routed sets. Global task-queue driven producer/consumer overlap. |

## Variants (build_cutedsl.py group: `nvfp4_fused_moe`)

12 active variants: 6 activations x 2 backends, all using MMA N-tile 128.

The C++ runner currently dispatches n128. Add a new N-tile dispatch axis only
after rebuilding the artifact pack with matching wrapper symbols and validating
accuracy/perf across the full matrix.

| Name | Backend | Activation |
|---|---|---|
| `nvfp4_fused_moe_decode_identity_n128` | decode | identity |
| `nvfp4_fused_moe_decode_silu_n128` | decode | silu |
| `nvfp4_fused_moe_decode_swiglu_n128` | decode | swiglu |
| `nvfp4_fused_moe_decode_gelu_n128` | decode | gelu |
| `nvfp4_fused_moe_decode_relu2_n128` | decode | relu2 |
| `nvfp4_fused_moe_decode_geglu_n128` | decode | geglu |
| `nvfp4_fused_moe_prefill_identity_n128` | prefill | identity |
| `nvfp4_fused_moe_prefill_silu_n128` | prefill | silu |
| `nvfp4_fused_moe_prefill_swiglu_n128` | prefill | swiglu |
| `nvfp4_fused_moe_prefill_gelu_n128` | prefill | gelu |
| `nvfp4_fused_moe_prefill_relu2_n128` | prefill | relu2 |
| `nvfp4_fused_moe_prefill_geglu_n128` | prefill | geglu |

All variants target SM120/SM121 only.

## Artifact Development

If you modify this kernel or its registry entries, manually regenerate the
`nvfp4_fused_moe` group before running CMake. Otherwise, CMake uses the matching
prebuilt tarball by default. Follow the shared
[CuTe DSL kernel development workflow](../README.md#cute-dsl-kernel-development-workflow)
for the supported Docker and local-venv commands, dependency versions,
cross-compilation, artifact layout, and CMake configuration.

## CuTe DSL SM121 Support

`nvidia-cutlass-dsl==4.7.0` supports the `sm_121a` architecture used by
DIGITS/GB10. For SM121 fused MoE builds, `build_cutedsl.py` sets
`CUTE_DSL_ARCH=sm_121a` for this group so the generated image targets SM121
directly instead of relying on an SM120-compatible cubin.

## Manual export (single variant)

```bash
cd kernelSrcs
python nvfp4_fused_moe_cutedsl/export_decode_kernel.py \
    --activation swiglu \
    --mma_tiler_n 128 \
    --output_dir /tmp/staging \
    --file_name nvfp4_fused_moe_decode_swiglu_n128 \
    --function_prefix nvfp4_fused_moe_decode_swiglu_n128 \
    --verbose
```

## CMake integration

`cmake/CuteDsl.cmake` auto-detects the `nvfp4_fused_moe` group from
`metadata.json` and sets `CUTE_DSL_NVFP4_FUSED_MOE_ENABLED` on the
link targets. C++ runner code should be guarded by:

```cpp
#ifdef CUTE_DSL_NVFP4_FUSED_MOE_ENABLED
#include "cutedsl_nvfp4_fused_moe_all.h"
// ... use generated _Kernel_Module_t / _wrapper symbols ...
#endif
```

## Attribution

The two kernel backends are ported from the
[`b12x`](https://github.com/lukealonso/b12x) kernel library by Luke Alonso:

| This repo | b12x origin |
|---|---|
| `moe_decode_kernel.py` | `b12x/moe/fused/static.py` (`MoEStaticKernel`) |
| `moe_prefill_kernel.py` | `b12x/moe/fused/dynamic.py` (`MoEDynamicKernelBackend`) |

The core compute body (FC1 → activation → quant → FC2 → scatter), the
queue-driven producer/consumer model, and the resident-grid barrier scheme
are derived from that work. Local changes include: additional activation
variants (`identity`, `gelu`, `swiglu`), SM121 artifact generation, and
integration with the TRT Edge-LLM plugin system.

## TensorRT plugin

The FP16 variants of this kernel family are wrapped as a TensorRT plugin at
[`cpp/plugins/nvfp4MoePluginGeforce/`](../../cpp/plugins/nvfp4MoePluginGeforce/).
See that plugin's
[`README.md`](../../cpp/plugins/nvfp4MoePluginGeforce/README.md) for the
supported-shapes contract and integration instructions.
