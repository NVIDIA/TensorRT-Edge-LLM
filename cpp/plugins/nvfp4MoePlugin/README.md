# Nvfp4MoePlugin

TensorRT plugin that wraps the SM100/SM101/SM110 decomposed split FC1/FC2
NVFP4 MoE kernel in
[`kernelSrcs/nvfp4_moe_cutedsl/`](../../../kernelSrcs/nvfp4_moe_cutedsl/).

This plugin is the **SM100/SM101/SM110** counterpart of
[`NvFP4MoEPluginGeforce`](../nvfp4MoePluginGeforce/nvfp4MoePluginGeforce.cpp),
which targets **SM120 / SM121 (consumer Blackwell)** via the fused
CuTe DSL path.

The two plugins share the same 11-input / 1-output ONNX surface and
attribute set, but consume **different on-disk weight layouts**:

* `Nvfp4MoePlugin` (this plugin) expects FC1 packed as the 64-row
  up/gate interleave
  ``[up_chunk(64), gate_chunk(64), up_chunk(64), ...]`` along the M
  axis (the layout the split FC1 kernel reads natively).
* `NvFP4MoEPluginGeforce` expects FC1 packed as the plain
  ``[up_all, gate_all]`` concat along the M axis (the layout the SM12x
  fused CuTe DSL kernel reads natively).

The Python export side picks the plugin and the matching weight repack
based on the target arch.

## Supported shapes

The split FC1/FC2 pack is specialized to `E=128` and `top_k<=8`;
hidden size and intermediate size stay runtime dimensions subject to the
alignment contract below. The n128 tactic is selected; n256 artifacts
may be generated but are not dispatched yet.

| Parameter | Supported values |
|---|---|
| `io_dtype` | FP16 |
| `SM` | 100, 101, 110 |
| `activation_type` | swiglu, relu2 |
| `routing_mode` | 0=softmax top-k (Qwen3), 1=sigmoid grouped top-k (Nemotron-H) |
| `backend` | decode + prefill (`auto` picks decode when `num_tokens*top_k` is small) |
| `hidden_size` (H) | `H > 0 && H % 128 == 0` |
| `moe_inter_size` (I) | `I > 0 && I % 64 == 0`, `FC1_N % 128 == 0` |
| `num_experts` (E) | `E == 128` |
| `top_k` | `0 < top_k <= 8` |

The plugin's `configurePlugin` enforces the divisibility and alignment
rules, emitting a clear error when they are violated.
`CuteDslNvfp4MoeSm110Runner::canImplement` is the authoritative source.

## Files

- [`nvfp4MoePlugin.h`](nvfp4MoePlugin.h) / [`nvfp4MoePlugin.cpp`](nvfp4MoePlugin.cpp) — the `IPluginV3` implementation.
- [`../../kernels/moe/nvfp4_cutedsl/cuteDslNvfp4MoeSm110Runner.{h,cpp}`](../../kernels/moe/nvfp4_cutedsl/)
  — the SM100/101/110 AOT-module owner: per-variant lazy loading,
  shape checks, workspace layout, and split FC1 / FC2 dispatch. Loaded
  modules remain resident for process lifetime.
- [`../../../kernelSrcs/nvfp4_moe_cutedsl/README.md`](../../../kernelSrcs/nvfp4_moe_cutedsl/README.md)
  — split FC1/FC2 kernel variants, AOT build flow, and data-layout notes.

## ONNX input surface

11 inputs, 1 output (identical layout to `NvFP4MoEPluginGeforce`; only
the plugin layer name and the FC1 packing convention differ):

```
router_logits      fp32    [T, E]         # pre-softmax; plugin applies moeTopkSoftmax
hidden_states      fp16    [B, S, H]      # per-token NVFP4 quant done inside the kernel
fc1_qweights       int8    [E, N1, H/2]   # N1 = 2*I (swiglu) or I (identity); 64-row up/gate interleave
fc1_blocks_scale   int8    [E, m_tiles_1, k_tiles_1, 32, 4, 4]
fc1_alpha          fp32    [E]
fc2_qweights       int8    [E, H, I/2]
fc2_blocks_scale   int8    [E, m_tiles_2, k_tiles_2, 32, 4, 4]
fc2_alpha          fp32    [E]
input_global_scale       fp32    [E]
down_input_scale         fp32    [E]
e_score_correction_bias  fp32    [E]     # router correction bias; zeros for softmax-topk
-> output                fp16    [B, S, H]
```

Block scales use the contiguous physical CuTe DSL NVFP4 layout
`[E, m_tiles, k_tiles, 32, 4, 4]`.

## Artifact Development

If you modify the associated kernel or its registry entries, manually
regenerate the `nvfp4_moe` artifact before building this plugin. Otherwise,
CMake uses the matching prebuilt tarball by default. Use the shared
[CuTe DSL kernel development workflow](../../../kernelSrcs/README.md#cute-dsl-kernel-development-workflow)
for the Docker or local-venv build and the subsequent CMake configuration.

CMake auto-defines `CUTE_DSL_NVFP4_MOE_ENABLED` on the plugin target.
Without it, the plugin still compiles but every `enqueue` call returns
an error (the plugin creator still registers, so deserialize paths work
in build-only environments).

## Retargeting other shapes

For `(H, I)` changes that still match the alignment contract in
[Supported shapes](#supported-shapes) (`E=128`, `top_k<=8`, `H % 128 == 0`,
`I % 64 == 0`, `FC1_N % 128 == 0`): no rebuild needed. Configure the plugin
with the desired tuple.

If a new MMA N-tile granularity is needed (e.g. `n64` / `n512`), add a
new `KernelVariant` row to `build_cutedsl.py`, add the matching static
modules and the switch arm in
[`cuteDslNvfp4MoeSm110Runner.{h,cpp}`](../../kernels/moe/nvfp4_cutedsl/),
and rebuild.

## Validation

The split-path backend is covered by the
[SM100/101/110 runner smoke and accuracy tests](../../../unittests/cpp/kernels/moe/nvfp4MoeCuteDslSm110Tests.cu).
Avoid validating production routing by instantiating Python-only helper
modules directly; model integration should be tested at the export path
that explicitly emits `Nvfp4MoePlugin` for an SM100/101/110 target.

### Thor sign-off checklist (runner-test equivalent)

1. Manually generate the `nvfp4_moe` artifact with the shared workflow.
2. Copy or mount the workspace on Thor.
3. Build the plugin with `-DENABLE_CUTE_DSL=nvfp4_moe -DCMAKE_CUDA_ARCHITECTURES=110a`.
4. Run `./build/unittests/unitTestKernelsMoe --gtest_filter='CuteDslNvfp4MoeSm110Test.*'`.
