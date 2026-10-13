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

## AttentionPlugin

**Functional Description**:
- Handles Rotary positional encoding, KVCache I/O, and MHA/GQA attention computation.
- Implements FP16 precision and covers all supported SMs of TensorRT Edge-LLM.
- Supports FP8 KV cache for improved memory efficiency with CUDA >= 11.8.
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
| `query_start_offsets` | `[N + 1]` | Physical row starts. MR1 uses entry padding, so entry `i` starts at `i * S_pad`; these are not compact logical prefix sums. |
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

Old batch-major engines are not supported by this ABI. Re-export the ONNX model
and rebuild the engine, and use export, build, plugin, and runtime artifacts
from a compatible revision. MR1 deliberately keeps the existing plugin creator
version: compatibility is enforced by ragged engine metadata, bindings, and
tensor descriptors, not by a dual-ABI plugin implementation.

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

1. **Quantization Phase**: `tensorrt-edgellm-quantize` or a supported pre-quantized checkpoint stores linear layers in INT4 weights-only groupwise format with group size 128.
2. **Export Phase**: During ONNX model export, `tensorrt_edgellm` emits quantized matrix multiplication custom-op nodes for Int4GroupwiseGemmPlugin.
3. **Engine Construction**: The TensorRT engine builder identifies Int4GroupwiseGemmPlugin operations via registered plugin creators and integrates them into the optimized computation graph.
4. **Runtime Execution**: During inference, the Int4GroupwiseGemmPlugin executes quantized GEMM/GEMV operations as nodes within the TensorRT engine's execution graph.

## Int4GroupwiseGemmPluginV2

`Int4GroupwiseGemmPluginV2` is the default INT4 groupwise GEMM backend. It shares the `Int4GroupwiseGemmPlugin` GEMM semantics — INT4 weights-only groupwise quantization, group size 128, FP16 accumulation, and symmetric (zero-point-free) quantization — but dispatches to AOT-compiled CuTe DSL (cuteDSL) W4A16 kernels instead of the AWQ CUDA-C++ kernels.

The fragment layout supports any positive output width. The final 128-channel
fragment is bounds-predicated by both the GEMM and GEMV kernels, so narrow
projections retain their native output shape.

**Configuration Parameters**:
- `N`: Output feature dimensions of the GEMM operation
- `K`: Inner dimensions of the GEMM operation
- `GroupSize`: Number of INT4 weight items corresponding to one scaling factor (currently supports 128 only)

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
