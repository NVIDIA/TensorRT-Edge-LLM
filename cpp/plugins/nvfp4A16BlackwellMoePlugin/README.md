# Nvfp4A16BlackwellMoePlugin

Thor (SM110) plugin for NVFP4-weight / FP16-activation routed MoE.
It replaces `Nvfp4A16MoePlugin` (Marlin) for Thor exports only; every other
platform keeps the Marlin plugin unchanged. The two plugins have different
weight layouts and distinct ONNX identities: an engine never carries both.

## Supported contract

| Field | Value |
|---|---|
| SM | 110 only (rejected at `configurePlugin` elsewhere) |
| hidden_states / output | FP16 `[T, H]`, `H % 128 == 0` |
| router_logits / expert_score_bias | FP32 `[T, E]` / `[E]` |
| `activation_type` | 4 (ReLU2) |
| `routing_mode` | 1 (sigmoid group top-k); `n_group`, `topk_group`, `norm_topk_prob`, `routed_scaling_factor` as in `Nvfp4A16MoePlugin` |
| `num_experts` / `top_k` | {128, 256, 512} / 1..32 |
| `moe_inter_size` | logical I, `I % 64 == 0`; FC1 N is padded to 128 inside the layout |
| weights | `fc{1,2}_qweights` int8 `[E, N/128, K/64, 128, 32]`, `fc{1,2}_block_scales` int8 `[E, N/128, K/64, 128, 4]`, `fc{1,2}_global_scales` fp32 `[E]` (`BLACKWELL_MOE_N128_K64_V1`; each 32-byte code row carries the TMA 32B swizzle image: rows 4-7 of every 8 swap their 16-byte halves) |
| `layout` | 1 (`BLACKWELL_MOE_N128_K64_V1`) |
| `backend` | 0 auto, 1 force decode kernels, 2 force grouped tcgen05 GEMM |
| `max_routed_rows` | padded permuted-row capacity, 0 = resolve from the profile (`T*top_k + E*127`, rounded to 128) |

## Export

`python -m tensorrt_edgellm.scripts.export --target-sm 110` exports of
Nemotron 3.5 Lightning (ModelOpt `W4A16_NVFP4` routed experts) emit this
plugin: `NemotronHMoEMLP._prepare_for_export_a16` stacks the experts with
`repack_nvfp4_a16_blackwell_moe_experts` and `forward` calls
`torch.ops.trt_edgellm.Nvfp4A16BlackwellMoePlugin` with the logical
`moe_inter_size` (1856, not the Marlin-padded 1920), `backend=0` and
`max_routed_rows=0`; the dynamo translation pins `layout=1`, as the dense
`Nvfp4A16BlackwellGemmPlugin` translation does. Every other `--target-sm` value, and an omitted
target, keeps the Marlin `Nvfp4A16MoePlugin` export unchanged. The route is
`tensorrt_edgellm.models.ops.use_blackwell_nvfp4_a16_moe()`, which reads the
same explicit target as the dense `Nvfp4A16BlackwellGemmPlugin` selector and
never probes the export host GPU; it is latched at export preparation and
re-checked in `forward`. `tensorrt_edgellm/onnx/export.py` keeps inputs 4, 7
and 8 (`fc1_global_scales`, `fc2_global_scales`, `e_score_correction_bias`)
FP32 through the initializer downgrade.

## Execution

* `T = 1` (decode): warp-per-token sigmoid top-k routing (contracts with
  `n_group > 1` or `E > 512` use the shared `moeSigmoidGroupTopk` instead) ->
  `decodeFc1Kernel` (dequant GEMV over the routed expert, split-K 2) + reduce
  (alpha, ReLU2) -> `decodeFc2Kernel` (per-slot alpha * router weight, fp32
  accumulation, split-K 8) + reduce: 5 GPU ops, deterministic. FC1 is
  launch-bounded to two CTAs per SM and FC2 to three; routing is a separate
  kernel because fusing it into every FC1 CTA cost 131 registers (one CTA per
  SM) and ~40% of FC1's streaming bandwidth inside the engine. The decode
  workspace is sized for the largest FC1 split-K the benchmark override
  (`EDGELLM_MOE_DECODE_FC1_SPLITK`) can select, so the size TensorRT records at
  build time never depends on the environment.
* **Programmatic Dependent Launch**: every kernel above issues `griddepcontrol.wait`
  before its first read of data its immediate predecessor produced (and before
  its first global write), immediately followed by
  `griddepcontrol.launch_dependents`; the runner launches them with
  `cudaLaunchAttributeProgrammaticStreamSerialization` (CUDA kernels through
  `cudaLaunchKernelEx`, the grouped GEMMs through the AOT wrapper's `enable_pdl`
  argument) so each kernel's prologue overlaps the previous kernel's tail. On by
  default; `EDGELLM_ENABLE_PDL=0` (read once, before the first enqueue) disables
  it for an A/B, the same knob as `Nvfp4MoePlugin`. The shared grouped-routing
  kernels used for `n_group > 1` carry no wait and are launched without the
  attribute, so that contract simply serializes. Every kernel triggers right
  after its wait (dependents are scheduled once all CTAs of the primary have
  started, i.e. during its last wave) and does the work that does not depend
  on its immediate predecessor before the wait: a kernel's inputs that were
  produced two or more launches earlier are complete and visible when it
  starts, because its predecessor passed its own wait before triggering. So
  `decodeFc1Kernel` stages the activation row (previous layer) before the wait
  and looks the expert up after; `decodeFc1ReduceKernel` reads the routing
  results first; the routing kernel pulls the bias into L2; the grouped GEMMs
  run their barrier/TMEM prologue before the wait. Nothing is written before a
  wait (TensorRT may still hand that memory to the running predecessor).
  `decodeFc2Kernel` can also look up its first slots' experts and `cp.async`
  their weight tiles for its K range into shared memory before waiting for
  FC1's output (`kDecodeFc2PrefetchSlots`, `EDGELLM_MOE_DECODE_FC2_PREFETCH`);
  it is sealed to 0 because the larger shared-memory carve-out stops FC2 CTAs
  from co-residing with the shared-expert GEMV TensorRT runs on its auxiliary
  stream, which cost more than the staging saved (decode step 11.44 / 11.51 /
  11.52 ms for 0 / 1 / 2 slots). Measured effect of PDL on Thor (CUDA-graph
  decode step, Nemotron 3.5 Lightning): consecutive plugin kernels start
  0.5-5.6 us before their predecessor ends (nsys), about 1% of the prefill step
  at ISL 2048 and within noise at decode.
* `T >= 2` (prefill and batched decode): warp-per-token sigmoid top-k routing -> single-CTA
  expert-contiguous tile layout (`permuted_idx`, `tile_group_idx`,
  `num_valid_tiles`) -> permuted-row gather (routed rows only; pad rows are
  never read into anything that survives) + output zeroing -> FC1 grouped GEMM
  (ReLU2 fused) -> FC2 grouped GEMM (router weight + scatter-add fused): 5 GPU
  ops. Token tile (per-expert padding granularity) tn8 up to 16 tokens, tn16 up
  to 32, tn32 up to 256, tn64 up to 2048, tn128 above. Contracts
  with `n_group > 1` fall back to the shared `moeSigmoidGroupTopk` +
  `buildLayoutGpu` pair for the first two ops.
* **CUDA-core kernels are NVRTC bundles** (the same plugin-JIT path as the XQA
  attention kernels and the dense `Nvfp4A16BlackwellGemmPlugin` GEMV): routing,
  tile layout, gather, decode FC1/FC2 and their split-K reduces live in
  `kernelSrcs/nvfp4A16BlackwellMoe/nvfp4A16BlackwellMoeKernels.cu`, embedded
  into the plugin library at build time and compiled once per layer in
  `configurePlugin` with every shape parameter (`E`, `top_k`, `H`, `I`, split-K,
  prefetch slots, dtype) baked in as `-D` macros. The cubin is serialized as the
  runtime-only `moe_jit_bundle` attribute, so deserialization never needs NVRTC;
  `clone()` (build-phase execution) and the runtime creator load it into a
  context-keyed module registry, and the kernels are launched through the driver
  API (`cuLaunchKernelEx` with the PDL attribute). The benchmark-only
  `EDGELLM_MOE_DECODE_*` overrides are therefore read at engine build and travel
  with the engine. Only the grouped tcgen05 GEMMs stay CuTe DSL AOT.
* AOT modules are loaded in `onShapeChange` (`Nvfp4A16BlackwellMoeRunner::prepare`);
  `enqueue` never loads modules or queries the device, so CUDA-graph capture is
  safe after one uncaptured warmup.

## Files

* `cpp/plugins/nvfp4A16BlackwellMoePlugin/nvfp4A16BlackwellMoePlugin.{h,cpp}`
* `cpp/kernels/moe/nvfp4A16BlackwellMoe/` — runner, dispatch policy, NVRTC
  compiler (`...JitCompiler`) and driver-API launcher (`...JitRunner`) of the
  CUDA-core kernels
* `kernelSrcs/nvfp4A16BlackwellMoe/nvfp4A16BlackwellMoeKernels.cu` — the
  CUDA-core kernels (routing, tile layout, gather, decode FC1/FC2, reduces),
  embedded for NVRTC
* `kernelSrcs/nvfp4_a16_blackwell_moe/` — CuTe DSL grouped GEMM (AOT group
  `nvfp4_a16_blackwell_moe`) and the on-board oracle
* `tensorrt_edgellm/checkpoint/repacking.py` — `repack_nvfp4_a16_blackwell_moe_experts`
* `tensorrt_edgellm/models/nemotron_h/modeling_nemotron_h.py` — export routing
  (`NemotronHMoEMLP`); `tensorrt_edgellm/models/ops.py`,
  `tensorrt_edgellm/onnx/{onnx_custom_schemas,dynamo_translations,export}.py` —
  custom op, ONNX schema, translation and FP32 initializer pins

## Validation

* `tests/python-unittests/test_nvfp4_a16_blackwell_moe_layout.py` — layout pin (CPU).
* `tests/python-unittests/test_nvfp4_a16_blackwell_moe_plugin.py` — engine build,
  serialization round-trip, decode/prefill numerics (SM110) and rejections.
* `unittests/cpp/kernels/moe/nvfp4A16BlackwellMoeRunnerTests.cu` — runner vs
  double-precision reference, CUDA-graph replay, dispatch policy.
* `unittests/cpp/kernels/moe/nvfp4A16BlackwellMoeJitTests.cpp` — JIT key
  validation, shared-memory budget, compile / bundle round trip (any CUDA 13
  host), module loading (SM110).
* `unittests/cpp/plugins/nvfp4A16BlackwellMoePlugin/` — creator contract.

## Thor sign-off checklist

1. `python kernelSrcs/build_cutedsl.py --kernels ALL --gpu_arch sm_110 --arch aarch64`
2. `cmake .. -DTRT_PACKAGE_DIR=/usr -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake -DEMBEDDED_TARGET=jetson-thor -DCUDA_CTK_VERSION=13.2 -DENABLE_CUTE_DSL=ALL -DBUILD_UNIT_TESTS=ON`
3. `./unittests/unitTestKernelsMoe --gtest_filter='Nvfp4A16BlackwellMoe*'`
4. `./unittests/unitTestPlugins --gtest_filter='Nvfp4A16BlackwellMoePlugin*'`
5. `pytest tests/python-unittests/test_nvfp4_a16_blackwell_moe_plugin.py`
