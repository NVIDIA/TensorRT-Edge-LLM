# TensorRT Plugins Guide

This documentation explains the usage of TensorRT plugins with TensorRT Edge-LLM and guides users to make further customizations.

## Overview

TensorRT plugins are custom operations that extend the functionality of the TensorRT core library through user-defined layer implementations. Within the TensorRT Edge-LLM framework, plugins provide specialized implementations for key large language model (LLM) inference operations that require optimizations beyond those available through standard TensorRT library deliverables.

### Plugin Architecture and Capabilities

TensorRT plugins are user-defined layers that implement the `IPluginV2DynamicExt` or `IPluginV3` interface. Note that TensorRT Edge-LLM is migrating all plugins to V3. TensorRT Plugins provide the following capabilities:

- **Feature Extension**: Extend functionality of existing TensorRT versions with new runtime and kernel level optimizations.
- **Modular Encapsulation**: Package complex computational logic into reusable components with configurable parameters.

### Current plugins

1. **AttentionPlugin**: Implements standard MHA (Multi-Head Attention) and GQA (Group Query Attention).
2. **Int4GroupwiseGemmPlugin**: Int4 weights-only groupwise GEMM and GEMV using the AWQ CUDA-C++ kernels (V1 plugin-packed weights).
3. **Int4GroupwiseGemmPluginV2**: Int4 weights-only groupwise GEMM and GEMV using the cuteDSL W4A16 kernels (fragment-layout weights). Default backend (see [Backend Selection](#backend-selection)).
4. **LowRankCandidateHeadPlugin**: Output head of EAGLE drafts with a low-rank candidate head, such as EAGLE-4 drafters; emits the top candidate token IDs and log-probabilities per row instead of full-vocabulary logits.

## AttentionPlugin

**Functional Description**:
- Handles Rotary positional encoding, KVCache I/O, and MHA/GQA attention computation.
- Implements FP16 precision and covers all supported SMs of TensorRT Edge-LLM.
- Supports FP8 KV cache on SM89, SM90, SM100, SM101, SM110, SM120, and SM121.
- Supports prefill (normal and chunked) stage causal attention.
- Supports vanilla decoding attention and tree decoding attention that is used by EAGLE and DFlash speculative decoding.
- Supports linear KVCache with equal capacity within one batch.
- Pads to maximum input sequence length within the batch for prefill execution.

**Configuration Parameters**:
- `num_q_heads`: Integer specification of query attention head count
- `num_kv_heads`: Integer specification of key-value head count (enables MQA/GQA configurations)
- `head_size`: Integer specification of per-head dimension size
- `enable_tree_attention`: Boolean flag to enable tree attention for speculative decoding implementations
- `kv_cache_type`: Data type for KV cache (FP16 or FP8)

**Decoder tensor contract (ragged ABI v1)**:

| Tensor | Shape | Meaning |
|---|---:|---|
| `PackedQKV` / `inputs_embeds` | `[T_exec, hidden]` | Token-major physical rows. QKV is packed on its last dimension. |
| `positions` | `[T_exec]` | Absolute position for each physical row. Padding uses `-1`. |
| `query_start_offsets` | `[N + 1]` | Physical row starts. Phases 1-7 use entry padding, so entry `i` starts at `i * S_pad`; phase 8 uses the compact prefix sums of the query lengths. |
| `query_lengths`, `past_lengths`, `sequence_lengths`, `attention_sequence_lengths` | `[N]` | Logical per-entry lengths. Padding changes `T_exec` and offsets, never these values or request progress. |
| `state_indices` | `[N]` | Maps current entries to resident KV/recurrent/conv pool rows. Current entry order is independent of resident slot order. |
| `kv_page_table` | `[N, 2, max_pages_per_sequence]` | Current-entry page-table view into the resident paged KV pools. |
| `RopeCosSin` and optional multimodal inputs | token-aligned with `T_exec` | RoPE, DeepStack/PLE, vision metadata, tree parent/depth, and similar decoder-side data follow the same physical rows and padding sentinels. |
| `logits_indices` | `[L]` | Physical token rows selected for logits; this is not a per-batch last-token tensor. |
| `execution_phase_marker` | `[phase_extent]` | Shape-only carrier; the INT32 payload is ignored. |

`N` is the active entry count, `S_pad` is the homogeneous step's physical
width, and `T_exec = N * S_pad`. `T_valid = sum(query_lengths)` can be smaller
than `T_exec`. Attention output is `[T_exec, hidden]`. Past/present KV and
recurrent/conv state use fixed resident pools and keep the same address; only
rows selected by `state_indices` participate in the step. Padded rows must not
read or update persistent state.

The phase marker extent has one shared meaning in Attention, GDN, causal Conv,
Mamba, and the runtime:

| Extent | Phase |
|---:|---|
| 1 | context prefill |
| 2 | context chunk |
| 3 | autoregressive decode |
| 4 | speculative draft proposal |
| 5 | speculative target verify |
| 6 | diffusion denoise |
| 7 | diffusion commit |
| 8 | compact mixed prefill and decode |

**Phase 8 (compact) contract**:

Phase 8 is implemented by Attention, GDN, causal Conv, and Mamba2 SSD
(`update_ssm_state`) layers. The runtime emits it for the compact steps of a
runtime session opened with compact execution; such a session still runs a step
in which every resident decodes as phase 3 (see
[In-Flight Batching](../software-design/in-flight-batching.md#execution-and-strategy)).

- **Rows**: the first `Np` entries are prefill sequences packed contiguously in
  rows `[0, T_p)`; each of the remaining `Nd = N - Np` entries owns one decode
  row, so `T_exec = T_p + Nd`. `query_start_offsets` holds the prefix sums of
  `query_lengths`. `Np` is the extent of the shape-only
  `context_sequence_count_carrier`, with `0 <= Np <= N`; `Np = 0` means
  `T_exec = N`. Every prefill sequence has at least one row; on the host a
  plugin can check only `T_p >= Np`.
- **Dispatch**: decode the ragged metadata with
  `decodeRaggedPluginMetadata(..., RaggedLayoutSupport::kPaddedAndCompact)`
  (the default `kPaddedOnly` rejects phase 8) and select the kernels by phase.
  A plugin decides from its attributes, static shapes, and the GPU whether it
  can run phase 8. `getWorkspaceSize()` and `enqueue()` must call the same
  decision function, with inputs that do not depend on the step's sequence
  count: the workspace reserved at build time covers phase 8 exactly when
  enqueue accepts it. `getWorkspaceSize()` also passes that decision to
  `reportCompactWorkspace()` on every call, because TensorRT may query a layer
  more than once; the builders write `builder_config.compact_execution` only
  when every such layer reported a reservation, and the runtime runs compact
  steps only on engines with that field. The decision function lives in `cpp/kernels/compactPhase/`,
  and the runtime's compact capability check (`compactLayerGaps`) calls the same
  function with the shapes from the engine's `config.json`; a new phase-8 plugin
  adds its function there and its layer kind to that check. An unsupported layer
  fails the phase 8 enqueue with `<plugin> '<layer>' (<shape>): compact execution is not
  supported for this layer on this GPU.` before it decodes metadata or writes
  any state. TensorRT passes `createPlugin()` an empty name at deserialization,
  so a phase-8 plugin serializes its layer name in the `layer_name` field
  (`layerNameField()`), restores it with `resolvePluginLayerName()`, and formats
  the layer with `describePluginLayer()`. The runtime treats such a refusal as
  permanent: it fails the step and disables compact execution for the runtime's
  life. The `RequestEngine` then continues with single-cohort steps or stops,
  depending on its scheduling mode (see
  [In-Flight Batching](../software-design/in-flight-batching.md#failure-handling)).
- **Decode rows**: rows `[T_p, T_exec)` must be bit-identical to phase 3. Emit
  them through the same decode function phase 3 uses (`enqueueDecodeRows`),
  offset by `T_p` rows and `Np` sequences. Outputs and state are written in
  place; only metadata that a kernel needs aligned may be staged in workspace.
  With `Np = 0` the kernel sequence equals phase 3.

Old batch-major engines are not supported by this ABI. Re-export the ONNX model
and rebuild the engine, and use export, build, plugin, and runtime artifacts
from a compatible revision. The plugin creator version is unchanged:
compatibility is enforced by ragged engine metadata, bindings, and tensor
descriptors, not by a dual-ABI plugin implementation.

The packaged experimental decoder `forward()` APIs are supported only through
their token-major `RaggedDecoderInputs` argument. Remaining batch-major rank-3
external APIs or plugin inputs in that package belong to standalone vision,
audio, or action encoder-domain components; they are not a legacy decoder
fallback. Decoder-internal `[T, heads, channels]` intermediates are valid: they
preserve token-major ownership and flatten to `[T, hidden]` before crossing the
decoder plugin boundary.


**Application Domains**:
- Transformer-based autoregressive language models that adopt standard MHA/GQA.

### Kernel Sources

Attention kernels are compiled into CUDA binaries. We provide the methods to produce CUDA binaries in `kernelSrcs/`.

**Kernel Libraries**:
- `fmha`: Canonical Context and ViT attention CuTe DSL AOT family built with
  `kernelSrcs/build_cutedsl.py`. It includes the FP16 FMHA-v2 kernels from
  `kernelSrcs/fmha_v2_cutedsl/fmha.py` on supported GPUs and the optimized
  Blackwell overlay from `kernelSrcs/fmha_cutedsl_blackwell/fmha.py` on
  SM100/SM101/SM110.
- `xqa`: Performant decoding attention kernels developed by NVIDIA. Implements normal decoding and tree-attention decoding.


### Integration Workflow

The AttentionPlugin integrates into the TensorRT Edge-LLM inference pipeline through the following stages:

1. **Export Phase**: During ONNX model export, `tensorrt_edgellm` emits attention custom-op nodes through TensorRT Edge-LLM ONNX translations.
2. **Engine Construction**: The TensorRT engine builder identifies plugin operations via registered plugin creators and integrates them into the optimized computation graph.
3. **Runtime Execution**: During inference, the AttentionPlugin executes as a node within the TensorRT engine's execution graph, with memory management handled by the TensorRT runtime.

## Int4GroupwiseGemmPlugin

**Functional Description**
- Implements A([M, K]) x B([K, N]) GEMM semantic where A is activation input, B is weights input.
- Supports INT4 weights-only groupwise quantization GEMM.
- Supports group size of 128.
- Accumulation is performed in FP16 precision for both GEMM and GEMV kernels.
- Implements symmetric quantization schema and zero-points is not supported.

**Configuration Parameters**:
- `N`: Output feature dimensions of the GEMM operation
- `K`: Inner dimensions of the GEMM operation
- `GroupSize`: Number of INT4 weight items corresponding to one scaling factor (currently supports 128 only)

**Input Tensors**:
- `GEMMInput`: Input activation tensor for the GEMM computation.
- `Int4Weights`: INT4 weights in the V1 plugin layout, packed into INT8 datatypes.
- `ScalingFactors`: Groupwise scaling factors.

**Output Tensors**:
- `GEMMOutput`: Result of the INT4 groupwise GEMM computation.

### Kernel Sources

A simplified kernel implementation is provided for this plugin. Evaluation indicates that this INT4 GEMM kernel achieves performance comparable to CUTLASS implementations on target production platforms (primarily Orin SKUs) with input sequence lengths (ISLs) of 2K to 3K tokens. Note that the GEMM kernel may not deliver sufficient performance for speculative decoding use cases with draft tree sizes of 64 to 128 tokens.

### Integration Workflow

The Int4GroupwiseGemmPlugin integrates into the TensorRT Edge-LLM inference pipeline through the following stages:

1. **Quantization Phase**: `trtedge quantize` or a supported pre-quantized checkpoint stores linear layers in INT4 weights-only groupwise format with group size 128.
2. **Export Phase**: During ONNX model export, `tensorrt_edgellm` emits quantized matrix multiplication custom-op nodes for Int4GroupwiseGemmPlugin.
3. **Engine Construction**: The TensorRT engine builder identifies Int4GroupwiseGemmPlugin operations via registered plugin creators and integrates them into the optimized computation graph.
4. **Runtime Execution**: During inference, the Int4GroupwiseGemmPlugin executes quantized GEMM/GEMV operations as nodes within the TensorRT engine's execution graph.

## Int4GroupwiseGemmPluginV2

`Int4GroupwiseGemmPluginV2` is the default INT4 groupwise GEMM backend. It shares the `Int4GroupwiseGemmPlugin` GEMM semantics — INT4 weights-only groupwise quantization, group size 128, and symmetric (zero-point-free) quantization — but dispatches to AOT-compiled CuTe DSL (cuteDSL) W4A16 kernels instead of the AWQ CUDA-C++ kernels. The GEMM accumulates in FP32; the GEMV keeps FP16 partial sums.

The fragment layout supports any positive output width. The final 128-channel
fragment is bounds-predicated by both the GEMM and GEMV kernels, so narrow
projections retain their native output shape.

**Configuration Parameters**:
- `N`: Output feature dimensions of the GEMM operation
- `K`: Inner dimensions of the GEMM operation
- `GroupSize`: Number of INT4 weight items corresponding to one scaling factor. The V2 runtime supports multiples of the 64-wide K tile, including 64 and 128.

**Input Tensors**:
- `GEMMInput`: Input activation tensor for the GEMM computation.
- `Int4Weights`: INT4 weights in the cuteDSL fragment layout, packed into INT8 datatypes.
- `ScalingFactors`: Groupwise scaling factors.

**Output Tensors**:
- `GEMMOutput`: Result of the INT4 groupwise GEMM computation.

### Kernel Sources

The plugin dispatches to AOT-compiled cuteDSL W4A16 GEMM kernels. The candidate kernel variants are generated by `kernelSrcs/build_cutedsl.py` (the `int4_fp16_gemm` group) into a static library, and the TensorRT plugin-V3 autotuner selects the best variant per problem shape at engine build time. A CUDA-core GEMV kernel serves the small-M (decode) regime, consuming the same fragment-layout weight buffer as the GEMM. The ONNX path prepares this layout during export; the ONNX-less builder can prepare it once during runtime initialization from the original checkpoint. Neither path repacks weights during inference.

### Backend Selection

Both plugins are registered and coexist. The `--int4-gemm-plugin-version` export CLI argument selects which one an INT4 checkpoint is exported for (the quantized checkpoint itself is backend-agnostic):

- `2` (default): `Int4GroupwiseGemmPluginV2` (cuteDSL fragment weights).
- `1`: `Int4GroupwiseGemmPlugin` (V1 plugin-packed weights) — legacy fallback.

The same selector drives both the weight repack (`checkpoint/repacking.py`) and the emitted custom op (`models/linear.py`), so the exported ONNX node and its weight layout always agree. The selected backend is logged once per export run.

## LowRankCandidateHeadPlugin

The ONNX export emits the `trt_edgellm::LowRankCandidateHeadPlugin` node as the
output head of an EAGLE draft that has a low-rank candidate head, such as the
[EAGLE-4](../../user_guide/examples/speculative-decoding.md#eagle-4) drafters.

**Functional Description**:
- Projects each hidden row to rank `r` (`s = down_proj h`, accumulated in FP32
  and stored as FP16) and scores every vocabulary token with `up_proj s`.
- Keeps the `num_candidates` (`Kc`) highest-scoring tokens; equal scores keep
  the smaller token ID and NaN scores rank last.
- Rescores the `Kc` candidates with exact FP32 dot products against their
  `lm_head` rows and outputs the log-softmax over those `Kc` logits. A
  non-finite logit sorts last with log-probability negative infinity.
- Results are deterministic and independent of the other rows in the batch.
- Supports FP16 inputs only and requires SM80 or newer.
- `enqueue()` allocates no device memory and keeps no state between calls, so
  the draft engine's CUDA graphs can capture it.

**Configuration Parameters**:
- `num_candidates` (`Kc`): ONNX attribute, the number of candidates per row, in
  `[1, 256]` and at most the vocabulary size `V`.
- `hidden_size`, `rank`, `vocab_size`: Read from the weight shapes at engine
  build time and serialized.
- `max_rows`: The largest row count `R` of the optimization profiles,
  serialized.
- `workspace_bytes`: The workspace the plugin needs for 64 rows on the build
  GPU, serialized. Every optimization profile reserves this size.

**Input Tensors**:
- `hidden`: `[R, H]` FP16 final-norm hidden states. `H` is a positive multiple
  of 8.
- `down_proj`: `[r, H]` FP16 low-rank factor. `r` is a multiple of 16 in
  `[16, 256]`.
- `up_proj`: `[V, r]` FP16 low-rank factor, one row per vocabulary token.
- `lm_head`: `[V, H]` FP16 output projection.

The weight shapes must be the same in every optimization profile.

**Output Tensors**:
- `support_ids`: `[R, Kc]` INT32 full-vocabulary token IDs, sorted by exact
  logit (descending), then token ID (ascending).
- `support_log_probs`: `[R, Kc]` FP32 log-probabilities of `support_ids`.

**Rows and Workspace**:
- `R` rows run in passes of at most 64 rows; each pass reads `up_proj` once.
- The plugin's first launch on a GPU picks its configuration for that GPU. If
  that GPU's preferred configuration needs more workspace than
  `workspace_bytes` because the build GPU had fewer SMs, less shared memory, or
  a smaller L2 cache (for example, a Jetson power mode with fewer GPU TPCs, or
  a smaller MIG instance), the plugin uses the first of a fixed sequence of
  smaller configurations that fits and logs the sizes at the INFO level. The
  outputs do not depend on the configuration; only the speed does.
- If `workspace_bytes` is below the smallest of those configurations, every
  `enqueueV3` call that runs the plugin fails before the plugin launches
  anything. The error states the size that the GPU needs. Rebuild the engine on
  that GPU, in the power mode it runs in.
