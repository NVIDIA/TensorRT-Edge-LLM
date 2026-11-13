<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Safety Runtime source boundary

The production Safety Runtime is built as the standalone `edgellmSafetyRuntime`
target. Its source and link closure is independent of the Standard Runtime
aggregate targets. Attention plugins and kernels are linked through the dedicated
`NvInfer_edgellm_plugin_safe` shared library.

The implementation is organized into these source areas:

- `runtime/`: execution ownership, status values, and error conversion.
- `runtime/model_inference/`: model-facing stage and callback types.
- `runtime/resources/`: host containers, IO allocations, and stream ownership.
- `trt_runner/`: TensorRT Safety/DLI graph preparation and execution.
- `plugins/`: Safety attention wrappers and plugin registration.
- `kernels/`: selected attention kernels and ahead-of-time (AOT) module loading.

Target capabilities are mapped to sources, dependencies, and verification gates
in `support_index.yaml`. This descriptive inventory is not consumed by the
runtime. Target model scope and required verification are recorded in the
`capabilities` entries. Model support and Safety qualification are established
only after the corresponding gates have passed. Standard sources may be used
only when listed individually under `reused_standard_sources` with their
dependency closure;
Safety-owned copies are listed under `copied_standard_sources`.

## Fixed-profile execution lifecycle

`TRTSafetyRunner` and `TRTResourceManager` are owned by `SafetyRuntime`. Each
lightweight-executor (LWE) graph is owned by a `TRTSafetyGraphExecutor`.
Graph execution is provided through LWE's public
`ITRTGraph` and `ISafeRecorder` interfaces with the Safety attention plugin
registry and module owner. Declared stages, slots, and fixed profiles are
executed through this interface. Model loading, tokenization, generation, and
the model-level request flow are outside this interface.

One CUDA context is created and kept current by the application. All calls are
serialized, and a runtime recorder and a process-lifetime registry recorder are
supplied by the application.
Construct `SafetyRuntime` with those borrowed recorders, then:

1. Call `initialize(spec)` with trusted serialized engines, their absolute
   matched companion-library paths, and the declared stage/slot/profile mappings.
   All executor modules are loaded before graph setup or IO/stream allocation.
   Set `SafetyGraphConfig::enableCudaGraphCapture` for each graph that should
   use prepared CUDA Graph execution; the default is direct execution.
   One logical main-stream ID per executor can be assigned through
   `SafetyRuntimeInitSpec::executorMainStreamIds`. For two executors sharing a
   catalog stream, use `{0, 0}`. That stream is created after module loading and
   owned by the resource manager until every borrowing graph is destroyed.
   CUDA's default stream is not selected by logical ID `0`. A separate main
   stream is created for each executor when the ID vector is empty.
2. Inspect `contexts()` and call `completeContext()` for each context declaring
   model-local resources. Caller-owned buffers are borrowed by these views;
   names and minimum capacities must match the declaration.
3. Call `prepare(warmup, userData)`. Known-valid inputs are supplied by the
   callback on each context's explicit main stream. Every declared stage/profile
   is executed during Initialization before the modules are sealed for Operational
   use. With capture enabled, each profile is first executed directly. Its enqueue
   is then captured without synchronization inside capture, the CUDA Graph is
   instantiated and uploaded, and its first replay is validated. Warmup inputs
   must remain valid for that additional execution. The capture is reused by
   later warmups.
4. Call `runLMStage(stage, slot, outputs, prepareInputs, userData)` with an
   `LMStageId` and `LMStageOutputs`. Exactly one declared profile is required for
   that stage/slot. Its optional `LMStagePrepareCallback` is invoked on every
   request after context selection, including repeated decode calls. Preallocated
   payloads are updated on the context's main stream without allocation or
   rebinding. Execution and output publication are permitted only after successful
   preparation and health inspection. Work is enqueued directly or through the
   prepared CUDA Graph, synchronized through LWE, and checked through its pinned
   host error bitmap before `outputs.context` is published. Assign a separate
   stage/slot entry to each profile exposed from an executor with multiple
   admitted profiles.
5. After healthy completion, call `deinitialize()` in Deinit Preparation. CUDA
   Graph executables and captured graphs are destroyed before their LWE graph
   and modules; all borrowing graphs precede IO buffer and stream release.

Explicit nondefault, nonblocking `cudaStream_t` streams are created and managed
by `TRTResourceManager` during Initialization. Main streams are shared according
to `executorMainStreamIds`; fixed auxiliary streams are owned per executor.
The handles are borrowed by LWE and reused during Operational execution.
Streams are released only during healthy Deinit Preparation after all borrowing
graphs have been destroyed.

IO buffers are sized from the maximum physical descriptor byte count across
admitted profiles, with stable addresses shared across those profiles. This
storage is shared by stages and slots assigned to the same executor, and their
execution is serialized. Exact in-place IO alias groups are declared by executor
index and tensor names through `SafetyRuntimeInitSpec::ioAliases`.
An alias group is admitted only for GPU tensors
with identical data types, physical layouts, and byte extents across every
admitted profile. The engine and plugin must support the declared in-place
operation. Each group is allocated and freed once; aliases are not inferred
from tensor names, and partial overlap is not admitted.

Context metadata and borrowed model-local views are fixed throughout Operational
execution. Allocation, IO rebinding, stream creation, graph creation, capture,
instantiation, and upload are excluded from the request path. Capture objects are
owned by the executor and prepared once per declared profile.
Capture and replay failures are terminal, without direct-execution fallback.
Capture is experimental; only device-resident IO payloads are admitted.
Payload-bearing host IO is rejected during Initialization; zero-byte metadata
does not require device payload storage.
Enable capture only for engines whose GPU topology and launch parameters remain
fixed per profile. Mutable device inputs at the captured addresses are supported;
LWE host instructions are executed during capture and are not replayed, so
per-request changes to their results must not affect GPU execution.
The capture boundary is one fixed-profile LMStage execution.

Tensor views are cached during Initialization. A `SafetyStatus` is returned by
the context's `getTensorView(profile, name, out)` lookup, and a borrowed
`SafetyTensorView const*` is supplied through `out`. The tensor name, pointer,
data type, memory placement, IO mode, logical and physical dimensions, strides
and stride order, vectorization, profile byte size, and allocation capacity are
exposed in the view. Project-owned enums are used rather than TensorRT descriptors
and bindings. Strides are expressed in scalar components rather than bytes.
For a singleton-profile context, the profile argument can be omitted through
`getTensorView(name, out)`; an explicit profile is required for ambiguous
contexts. No allocation or SDK call is performed during lookup. The metadata is
fixed while payloads are updated through the borrowed pointer. A null pointer
is permitted for zero-byte metadata. Views are valid until `deinitialize()`
begins; output payloads are consumed only after successful execution.

Terminal state is latched by CUDA/DLI failures, recorder errors/overflow, and
asynchronous device error bits, including during Initialization and
Deinitialization. No CUDA/DLI work is performed by subsequent calls. Partial
resources are retained; there is no retry, fallback, automatic rollback, or
destructor-triggered cleanup.
Keep the owners and borrowed recorders alive until application containment and
process/context recovery complete. Argument/state rejection before SDK calls
is nonfatal. C++ container allocation is confined to Initialization. Native
execution does not establish QNX Safety library qualification or Safe CUDA
readiness/VM transitions.

The production implementations are compiled against genuine LWE headers for the
host lifecycle tests, with SDK/CUDA test doubles. No GPU or Safety runtime library
is required; engine or model compatibility is not established by these tests. Set
`TRT_PACKAGE_DIR` to the package root containing the matched public Safety
headers in `include/`. Run build commands from the repository root:

```bash
cmake -S tests/safety/runtime -B build-safety-runtime-host \
  -DTRT_PACKAGE_DIR="$TRT_PACKAGE_DIR"
cmake --build build-safety-runtime-host --parallel 8
ctest --test-dir build-safety-runtime-host --output-on-failure
```

For split header packages, also set `TRT_SAFE_BASE_INCLUDE_DIR` to the directory
containing `NvInferRuntimeBase.h`. Use matched TensorRT Safety 11.3+ headers and
libraries for production builds; the factory requires a companion-library path.
RTX 5090 engine execution and DRIVE OS validation require separate real artifact
tests. The attention integration below currently admits SM80 and SM110a
artifacts; SM120 attention artifacts are not yet admitted.

The real-engine runner probe accepts `--cuda-graph` to exercise capture and
replay with its matched engine and companion library:

```bash
safetyRunnerSmoke matching-safety-engine.plan result.json --cuda-graph
```

Safe CUDA capture API availability and lifecycle qualification require the
matched target SDK and separate DRIVE OS validation.

## Safety attention artifacts and builds

TensorRT 11.3 or later is required. Five-input, batch-one FP16 decoder graphs
with batched QKV and contiguous KV cache `[1, 2, kv_heads, capacity, head_size]`
are accepted by the LLM plugin.
Token-major QKV and paged KV exports are not supported.

Select batch-one fixed profiles when using this plugin; a more general exported
batch dimension does not imply runtime support for larger batches. Pass the
built `<build-dir>/cpp/safety_inference/libNvInfer_edgellm_plugin_safe.so`
explicitly to the Safety builder through `EDGELLM_SAFETY_PLUGIN_PATH`.

Generate the attention artifacts before configuring either build. In a build-only
Python environment, install `nvidia-cutlass-dsl==4.7.0`, `cuda-bindings==13.3.1`
and a matching CuPy package. Safety-owned copies of the selected Standard kernels
are used for generation: Blackwell for Thor and Ampere for SM80 host CI. The
copied sources, device validation, and AOT launch adapters are recorded in the
artifact manifest.
Python and CuPy are build-only dependencies, not target-runtime dependencies.
Use architecture `80` for A30 host tests or `110a` for DRIVE Thor:

```bash
export SAFETY_CUDA_ARCH=80
export EDGELLM_SAFETY_FMHA_ARTIFACT_DIR="$PWD/build-safety-fmha"
python cpp/safety_inference/kernels/fmha/generate.py \
  --arch "$SAFETY_CUDA_ARCH" --output-dir "$EDGELLM_SAFETY_FMHA_ARTIFACT_DIR"
```

For D7QS, also pass `--cuda-compiler /path/to/target-sdk/bin/nvcc` (or set
`EDGELLM_SAFETY_CUDA_COMPILER`) when generating artifacts. Unchanged CuTe PTX is
assembled into native cubins by the target SDK; CuTe's bundled assembler output
is not admitted by the QNX Safety build. The SDK compiler version is recorded
in the manifest. Retained PTX is build evidence only and is not deployed.

Configure a development build containing both runtimes with:

```bash
cmake -S . -B build -DENABLE_SAFETY_RUNTIME=ON \
  -DTRT_PACKAGE_DIR=/path/to/TRT-with-safety-runtime \
  -DCMAKE_CUDA_ARCHITECTURES="$SAFETY_CUDA_ARCH" \
  -DEDGELLM_SAFETY_FMHA_ARTIFACT_DIR="$EDGELLM_SAFETY_FMHA_ARTIFACT_DIR"
cmake --build build --target edgellmSafetyRuntime llm_build visual_build
```

This combined mode is used by the x86 native Safety Builder flow and the D7L
development proxy flow. Standard TensorRT and the ONNX parser are configured in
addition to the dedicated Safety Runtime target. `ENABLE_SAFETY_RUNTIME` is
propagated to the builder targets. `EngineCapability::kSAFETY` is selected by
`llm_build --safety` and `visual_build --safety`; fixed MSS profiles are required,
and the tuning method is selected from the remote configuration. Without
a remote URI, tuning is performed on the local GPU; this requires a compatible GPU,
CUDA and TensorRT builder, and is not tied to D7L. Supply
`--remoteAutoTuningConfig` or `EDGELLM_REMOTE_AUTOTUNING_CONFIG` for remote
auto-tuning in CPU-only builder mode. A malformed URI is rejected, not treated
as a request for local tuning. Credentials are not written to the generated
engine configuration.

The deployment target is not classified as proxy or Safety OS by the builder.
Either a QNX Standard proxy or a Safety OS environment can be used for remote
tuning. The appropriate build method and target setup are selected by the
customer; safety qualification is not established by `kSAFETY` or remote tuning
alone. Only the build method is recorded in the generated
`builder_config.remote_auto_tuning`.

The `--safety` decoder exports are consumed by the LLM builder: batched `inputs_embeds`,
packed QKV, and contiguous KV cache `[B, 2, Hkv, capacity, D]`, with the
five-input `trt::AttentionPlugin` v1 interface. Token-major inputs and page tables
are rejected, including in fully static graphs. FP8 linear-layer Q/DQ does not
change this contract: attention and KV remain FP16, and RoPE remains FP32.
RoPE shape `[batchSize, kvCacheCapacity, rotaryDim]` is selected by each MSS profile;
the exporter's axes are not restricted by this builder profile policy.
The matching host checker and target runtime plugin must implement this same
interface. Decoder inputs are supplied through the external embedding path.

TensorRT 11.3 or later is required for Safety builds. Networks are built through
`buildSerializedSafeNetwork()`, and both `llm_safety.engine` / `visual_safety.engine`
and the matching `.engine.so` companion library are saved. Deploy the pair together.
The engine, companion, configuration and copied sidecars are staged before
existing files are replaced, and restoration of the previous set is attempted
on publication failure. This is not crash-atomic;
output directories must not be shared with a concurrent build or consumed while
publishing artifacts. Backups are retained if rollback cannot complete.

For QNX remote builds, the host needs both the matching CUDA toolkit and a QNX
host compiler to generate the companion library. Configure the QNX toolchain's
`QNX_HOST` and `QNX_TARGET`, and select its compiler through TensorRT/Myelin's
`-kgen:package_engine_so:host_compiler` option in `__LUNOWUD`. A CPU-only native
builder container does not necessarily contain that QNX compiler.

For a local build on a compatible development machine:

```bash
visual_build --safety --onnxDir /path/to/visual-onnx \
  --engineDir /path/to/engines --mssProfile 512
```

Native-operator visual ONNX exports do not require a Safety plugin library.
Total image tokens, rather than tile counts, are specified by InternVL profiles.
Tokens per tile are derived from `vision_config.image_size`, `vision_config.patch_size`, and
the `downsample_ratio` recorded in exported `config.json`.
An explicit checkpoint ratio is preserved by the shared exporter; if absent,
0.5 is recorded by its sidecar writer. Token counts follow the exporter's pixel-shuffle
scale `int(1.0 / downsample_ratio)`, with each patch-grid dimension divided by
that scale.
For 448x448 tiles and 14x14 patches, ratios 0.5 and 0.25 yield 256 and 64 tokens
per tile respectively; MSS 512 selects two and eight tiles. Missing metadata,
non-divisible pixel-shuffle grids, partial-tile profiles, and inconsistent ONNX output shapes
are rejected. Re-export legacy configs that omit `downsample_ratio`.

For exports requiring the ViT plugin, supply `--safetyPluginLibrary` or
`EDGELLM_SAFETY_PLUGIN_PATH`; missing creators are not replaced with Standard plugins.

For a target package that contains only Safety Runtime artifacts, specify its
root separately:

```bash
cmake -S . -B build-safety -DENABLE_SAFETY_RUNTIME=ON \
  -DEDGELLM_SAFETY_ONLY_PACKAGE=ON \
  -DTRT_PACKAGE_DIR=/path/to/TRT-safety-only \
  -DTENSORRT_SAFE_ROOT=/path/to/TRT-safety-only \
  -DCMAKE_CUDA_ARCHITECTURES="$SAFETY_CUDA_ARCH" \
  -DEDGELLM_SAFETY_FMHA_ARTIFACT_DIR="$EDGELLM_SAFETY_FMHA_ARTIFACT_DIR"
cmake --build build-safety --target edgellmSafetyRuntime
```

Standard TensorRT, the ONNX parser, Standard Edge-LLM targets, and examples are
not configured in this mode. A matching platform CUDA compiler is required by
the attention kernels. D7QS additionally requires the licensed QNX toolchain and
`QNX_SAFE_PDK_ROOT` pointing to the matching DRIVE Safety PDK.

For functional validation on standard QNX, keep the runtime-only package scope
and set `QNX_SAFE_CUDA=OFF`, with standard QNX CUDA target paths and a matching
standard-QNX `libnvinfer_safe` package:

```bash
cmake -S . -B build-safety-qnx-standard \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/aarch64_qnx_toolchain.cmake" \
  -DENABLE_SAFETY_RUNTIME=ON -DEDGELLM_SAFETY_ONLY_PACKAGE=ON \
  -DQNX_SAFE_CUDA=OFF -DQNX_SAFE_PDK_ROOT= \
  -DCUDA_CTK_VERSION="$CUDA_CTK_VERSION" \
  -DQNX_CUDA_TARGET_ROOT="$QNX_CUDA_TARGET_ROOT" \
  -DCUDA_DIR="$CUDA_DIR" -DCUDA_TARGET_DIR="$CUDA_TARGET_DIR" \
  -DTENSORRT_SAFE_ROOT="$TENSORRT_SAFE_ROOT" \
  -DCMAKE_CUDA_ARCHITECTURES=110a \
  -DEDGELLM_SAFETY_FMHA_ARTIFACT_DIR="$EDGELLM_SAFETY_FMHA_ARTIFACT_DIR" \
  -DEDGELLM_BUILD_SAFETY_ENGINE_TESTS=ON
cmake --build build-safety-qnx-standard \
  --target edgellmSafetyRuntime safetyPluginLinkCheck safetyEngineSmoke safetyRunnerSmoke
```

Ordinary QNX CUDA is used by the probes without NvDVMS gates. Deploy their matching
standard-QNX engine companion and runtime libraries; driver stubs are link-only.
`QNX_SAFE_CUDA` defaults to `EDGELLM_SAFETY_ONLY_PACKAGE` when first configuring,
so Safe CUDA is selected for the D7QS runtime-only build. Use a fresh build
directory when changing this platform selection.

QNX builds default to SDK cross-GCC/G++ with a warning when `QNX_USE_QCC`
is unset. Set it to `0` to select GCC explicitly or `1` for licensed `qcc/q++`.
NVCC always uses cross-G++. Use a fresh build directory when changing compilers.


### Source and link boundary gate

`support_index.yaml:boundary_gate` is the executable admission policy. The
configured phase reads the CMake File API and compilation database, validates
the exact production-target closures, configured sources, and unsupported API
patterns, then records the accepted targets, objects, artifacts, link metadata,
tools, and source digests in `<build>/safety-boundary/plan.json`. The artifact
phase refuses policy or configured-evidence drift and checks archive members,
effective link inputs, dynamic dependencies, and unresolved symbols with the
configured target toolchain. It writes `<build>/safety-boundary/report.json`.
Cross-built artifacts are never loaded or passed to `ldd`.

The implementation lives in `scripts/safety_checker`, with phase orchestration
in `check_safety_runtime_boundary.py`. `common.py` owns the shared records,
policy loading, bounded reads, tool execution, and reporting.
`build_metadata.py` normalizes the CMake File API, compilation database, tool
cache, and configured plan. `source_checks.py` validates target closures,
source paths, compiled headers, and prohibited APIs. `artifact_checks.py`
validates archives, final links, dynamic dependencies, and undefined symbols.


Each failure is printed as a stable single-line record with `check`, `target`,
`kind`, and `actual` fields. The JSON report records the phase, pass/fail
status, and the same sorted findings; successful checks do not emit positive
observations. The configured plan separately retains the accepted evidence for
artifact-phase drift detection. `llm_inference_safe` remains optional only
while `runtime_consumed` is `false`; it must be added to the build and made
required before the Safety Runtime is consumed.

## Caller ownership and lifecycle

Keep an executor-owned `SafetyAttentionModules` object per CUDA context and keep
that context current. Call `modules.load()` immediately after context creation,
before CUDA runtime initialization or graph setup. All application modules are
loaded before cudart is initialized for the driver-created context. Safe CUDA
API admission and lifecycle qualification require the matched target SDK manual
and separate target validation.

Plugins are registered and their borrowed module owner is bound during graph
creation by `createSafetyGraph(modules, ...)`. Entry points and device properties
are prepared once per owner by `initResource()`; the same owner is retained by
clones. `SafetyAttentionModuleScope` is used around plugin creation and building
by direct builder callers. Serialize these Initialization operations with other
registry operations; the temporary
creator routing is never consulted during execution. Keep the registry recorder
and plugin DSO alive until process exit, and each graph recorder longer than its graph.

Prepare fixed IO/workspace and warm up before `modules.seal()` and Operational.
Blackwell tensor maps are encoded during warmup; changed or unprepared bindings
are rejected after sealing. Complete `cudaSafeExNotifyNvDvmsTransitionToOperational()`
before the external VM transition. Check synchronous and asynchronous graph errors
before accepting results; failures are terminal, without retry or rollback. The
modules are released through `modules.release()` during healthy Deinit Preparation
after GPU completion and destruction of every borrowing graph/plugin. Implicit
CUDA cleanup is not performed by owners, and they cannot be reinitialized after
failure or release.

## Verification

Configure `tests/safety` separately with the TensorRT Standard and Safety headers,
CUDA architecture and AOT artifact directory to run contract, kernel and module
tests. Driver doubles are used by module lifecycle tests; QNX platform transitions
are not exercised.

The same C++ tests can be cross-built for standard QNX with matching CUDA and
Standard and Safety SDK headers; no TensorRT engine or runtime library is required. Set the
QNX toolchain variables described above and use standard, not Safe, CUDA paths:

```bash
cmake -S tests/safety -B build-qnx-tests \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/aarch64_qnx_toolchain.cmake" \
  -DCUDA_CTK_VERSION="$CUDA_CTK_VERSION" -DCUDA_DIR="$CUDA_DIR" \
  -DQNX_CUDA_TARGET_ROOT="$QNX_CUDA_TARGET_ROOT" \
  -DTRT_PACKAGE_DIR="$TRT_PACKAGE_DIR" -DCMAKE_CUDA_ARCHITECTURES=110a \
  -DEDGELLM_SAFETY_FMHA_ARTIFACT_DIR="$EDGELLM_SAFETY_FMHA_ARTIFACT_DIR" \
  -DEDGELLM_TEST_QNX_STANDARD=ON
cmake --build build-qnx-tests --target safetyPluginTests --parallel 8
cmake --install build-qnx-tests --prefix "$PWD/qnx-tests" --component safety-plugin-tests
```

Copy `qnx-tests/` to a standard-QNX Thor target. From that directory, run
`LD_LIBRARY_PATH="$PWD:${LD_LIBRARY_PATH:-}" ./safetyPluginTests --gtest_output=xml:results.xml`.
Use the target's installed standard CUDA libraries; do not deploy driver stubs.

`EDGELLM_BUILD_SAFETY_ENGINE_TESTS=ON` adds `safetyEngineSmoke` and
`safetyRunnerSmoke`; `safetyEngineBuild` is also provided by combined builds.
The same synthetic ViT engine is consumed by both probes, and its attention
outputs are validated against a CPU reference. The production runner, executor
and resource manager are exercised by `safetyRunnerSmoke` with two stage/slot
mappings, borrowed model-local resources, stable IO/stream views, and recoverable
argument rejection. Full-model inference is not validated by these probes.

The matching `engine.plan.so` is also written by the builder; deploy it beside
`engine.plan`. This matched companion library is loaded through an absolute path
by both probes. For example:

```bash
safetyEngineBuild engine.plan
safetyRunnerSmoke engine.plan runner-result.json
```

Native attention operations are used by the Safety InternVL visual exporter.
`ViTAttentionPlugin` is not exercised by its engine; that optional plugin path
is validated separately by the synthetic probe.

For QNX Safety, configure the standalone tests with
`EDGELLM_TEST_QNX_SAFE=ON`, `EDGELLM_SAFETY_ONLY_PACKAGE=ON`, Safe CUDA paths and
`QNX_SAFE_PDK_ROOT`. `safetyKernelSmoke` is built instead of the Standard
negative-test suite. D64/D128 visual ragged and causal contiguous attention are
checked against uniform attention references, including KV cache writes.
Nonzero-Q/K RoPE arithmetic and failure injection are covered by the Standard tests.

`safetyKernelSmoke result.json --external-lifecycle`,
`safetyEngineSmoke engine.plan result.json --external-lifecycle` and
`safetyRunnerSmoke engine.plan result.json --external-lifecycle` require an
exclusive external VM coordinator on D7QS and the `Nvdvms/ClientCommGetState`
ability. Start in INIT_DONE; after each `LIFECYCLE_READY` marker, the coordinator
advances to the requested phase: Operational after warmup, then Deinit Preparation
after synchronized execution. Each wait is bounded to 120 seconds. The harness
never changes VM state, retries execution or performs CUDA cleanup after SDK
failures.
After successful teardown the coordinator must shut down/reboot the VM, not
reinitialize CUDA. A timeout/failure needs platform-owner recovery; do not rerun
the harness in the same lifecycle. Use installed Safe CUDA libraries, never
Standard replacements or driver stubs. A real graph also requires a matching
Safety engine and runtime; the kernel smoke does not require an engine.

Safety CI build checks are manual and non-blocking; run them explicitly when
reviewing Safety changes. QNX board execution remains a separate validation step.
