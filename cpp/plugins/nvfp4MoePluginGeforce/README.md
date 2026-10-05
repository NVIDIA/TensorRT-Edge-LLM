# NvFP4MoEPluginGeforce

TensorRT plugin that wraps the fused NVFP4 MoE kernel in
[`kernelSrcs/nvfp4_fused_moe_cutedsl/`](../../../kernelSrcs/nvfp4_fused_moe_cutedsl/)
(CuTe DSL SM120 / SM121, decode + prefill backends).

This plugin is the **SM12x (consumer Blackwell)** counterpart of
[`Nvfp4MoePlugin`](../nvfp4MoePlugin/nvfp4MoePlugin.cpp), which targets
**SM110 (Thor)** via the split FC1/FC2 backend.

The two plugins share the same 11-input / 1-output ONNX surface and
attribute set, but consume **different on-disk weight layouts**:

* `NvFP4MoEPluginGeforce` expects FC1 packed as the plain
  ``[up_all, gate_all]`` concat along the M axis (the layout the SM12x
  fused CuTe DSL kernel reads natively).
* `Nvfp4MoePlugin` expects FC1 packed as the 64-row up/gate interleave
  ``[up_chunk(64), gate_chunk(64), up_chunk(64), ...]`` (the layout the
  SM110 split FC1 kernel reads natively).

The Python export side picks the plugin and the matching weight repack
based on the target arch.

## Supported shapes

The AOT-exported CuTe DSL kernel is **shape-polymorphic** in `I` / `E` /
`top_k`: the wrappers in
[`export_decode_kernel.py`](../../../kernelSrcs/nvfp4_fused_moe_cutedsl/export_decode_kernel.py)
and
[`export_prefill_kernel.py`](../../../kernelSrcs/nvfp4_fused_moe_cutedsl/export_prefill_kernel.py)
build `cute.Tensor` layouts from raw pointers and runtime `Int32`
extents, so one AOT binary handles any (I, E, top_k) tuple meeting the
alignment contract below.

One dim is a **compile-time variant axis** and requires a kernel rebuild
to change:
* The **MMA N-tile** currently dispatches n128. Add another dispatch axis
  after rebuilding the artifact pack and re-enabling n256 in
  `CuteDslNvfp4MoeRunner::selectMmaTilerN`.

The **hidden_size (K)** is runtime-shaped in the AOT wrapper. The only
constraint is `H > 0 && H % kHiddenSizeAlignment == 0`, where
`kHiddenSizeAlignment = kCuteDslTileK * kStaticAbStage = 128 * 2 = 256`
in the current build. That divisor guarantees the K-tile mainloop pipeline
drains cleanly with the AOT-baked `ab_stage = 2`. Validated for H in
`{1024, 2048}`; other multiples of 256 are expected to work by shape
polymorphism but should be accuracy-checked before production use.

| Parameter | Supported values |
|---|---|
| `io_dtype` | FP16 |
| `activation_type` | identity, silu, swiglu, gelu, relu2, geglu |
| `routing_mode` | 0=softmax top-k (Qwen3), 1=sigmoid grouped top-k (Nemotron-H) |
| `backend` | decode + prefill (`auto` picks decode when `num_tokens*top_k <= 640` else prefill; mirrors `select_sm120_moe_backend`) |
| `hidden_size` (H) | positive multiple of `kHiddenSizeAlignment` (= 256 = `kCuteDslTileK * kStaticAbStage`) |
| `moe_inter_size` (I) | `I > 0`, `I % kLevelTileN == 0` (128) |
| `num_experts` (E) | `E > 0` |
| `top_k` | `0 < top_k <= E` |

The plugin's `configurePlugin` enforces the hidden-size divisibility and
the alignment rules, emitting a clear error when they are violated.
`CuteDslNvfp4MoeRunner::canImplement` is the authoritative source.

### Decode vs prefill

Both backends' AOT binaries are shape-polymorphic in `(H, I, E, top_k)`
-- weights and per-expert metadata are passed as pointers + runtime
`Int32` extents, so TMA descriptors are rebuilt on every launch. The
difference is purely the per-enqueue execution pattern:

* **Decode** — resident-grid kernel with a block-wide barrier between
  route/pack and compute phases. Lower launch overhead; best when
  `num_tokens*top_k <= 640`.
* **Prefill** — global task-queue-driven producer/consumer overlap.
  Higher launch overhead but scales to hundreds of routed rows per
  expert; best for prefill workloads.

`backend=auto` applies the same cutover
(`kDecodePrefillCutoverRoutedRows = 640`) as
[`select_sm120_moe_backend`](../../../kernelSrcs/nvfp4_fused_moe_cutedsl/moe_dispatch.py).

## Files

- [`nvfp4MoePluginGeforce.h`](nvfp4MoePluginGeforce.h) / [`nvfp4MoePluginGeforce.cpp`](nvfp4MoePluginGeforce.cpp) — the `IPluginV3` implementation.
- [`../../kernels/moe/nvfp4_cutedsl/cuteDslNvfp4MoeRunner.{h,cpp}`](../../kernels/moe/nvfp4_cutedsl/) — the AOT-module owner: per-variant lazy loading, shape checks, workspace layout, and wrapper dispatch. Loaded modules remain resident for process lifetime. The runner is allocation-free on the enqueue path; the plugin owns the per-instance identity expert-id table (via `IGpuAllocator` in `attachToContext`) and threads it in through `CuteDslNvfp4MoeParams::weightExpertIds` / `globalToLocalExpertIds`.
- [`../../../kernelSrcs/nvfp4_fused_moe_cutedsl/README.md`](../../../kernelSrcs/nvfp4_fused_moe_cutedsl/README.md) — kernel variants, AOT build flow, and data-layout notes.

## ONNX input surface

11 inputs, 1 output (identical layout to `Nvfp4MoePlugin`; only the
plugin layer name and the FC1 packing convention differ):

```
router_logits      fp32    [T, E]         # pre-softmax; plugin applies moeTopkSoftmax
hidden_states      fp16    [T, H]         # primary token-major API; per-token NVFP4 quant in the kernel
fc1_qweights       int8    [E, N1, H/2]   # N1 = 2*I (swiglu/geglu) or I otherwise; plain [up,gate] concat
fc1_blocks_scale   int8    [E, m_tiles_1, k_tiles_1, 32, 4, 4]
fc1_alpha          fp32    [E]
fc2_qweights       int8    [E, H, I/2]
fc2_blocks_scale   int8    [E, m_tiles_2, k_tiles_2, 32, 4, 4]
fc2_alpha          fp32    [E]
input_global_scale       fp32    [E]
down_input_scale         fp32    [E]
e_score_correction_bias  fp32    [E]     # router correction bias; zeros for softmax-topk
-> output                fp16    [T, H]
```

Block scales use the contiguous physical CuTe DSL NVFP4 layout
`[E, m_tiles, k_tiles, 32, 4, 4]`.

## Artifact Development

If you modify the associated kernel or its registry entries, manually
regenerate the `nvfp4_fused_moe` artifact before building this plugin.
Otherwise, CMake uses the matching prebuilt tarball by default. Use the shared
[CuTe DSL kernel development workflow](../../../kernelSrcs/README.md#cute-dsl-kernel-development-workflow)
for the Docker or local-venv build and the subsequent CMake configuration.

CMake auto-defines `CUTE_DSL_NVFP4_FUSED_MOE_ENABLED` on the plugin target.
Without it, the plugin still compiles but every `enqueue` call returns an error
(the plugin creator still registers, so deserialize paths work in build-only
environments).

## Retargeting other shapes

For `(H, I, E, top_k)` changes that still match the alignment contract
(`H > 0 && H % 256 == 0`, `I % 128 == 0`, `0 < top_k <= E`): no rebuild
needed. Configure the plugin with the desired tuple and the runner will use
the n128 N-tile variant at launch.

If a new MMA N-tile granularity is needed (e.g. `n64` / `n512`), add a
new `KernelVariant` row per activation / backend to
`build_cutedsl.py`, add the matching static modules and the switch arm
in `selectMmaTilerN` / `DISPATCH_*_ACTIVATION` in
[`cuteDslNvfp4MoeRunner.{h,cpp}`](../../kernels/moe/nvfp4_cutedsl/),
and rebuild.

## Validation

There is no active Python unit-test entry for this plugin. Model
integration should be tested at the export path that explicitly emits
`NvFP4MoEPluginGeforce` for an SM12x target.

### SM12x sign-off checklist (runner-test equivalent)

1. Manually generate the `nvfp4_fused_moe` artifact with the shared workflow.
2. Build the plugin with `-DENABLE_CUTE_DSL=nvfp4_fused_moe`.
3. Run the full export → engine build → inference pipeline on a sm_120 /
   sm_121 host and verify the engine layer name binds to
   `NvFP4MoEPluginGeforce` (e.g. via `trtexec --dumpProfile`).
