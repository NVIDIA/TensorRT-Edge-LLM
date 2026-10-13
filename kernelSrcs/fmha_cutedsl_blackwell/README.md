# CuTe DSL FMHA Kernels (Blackwell SM100/SM101/SM110)

The `fmha` group always contains the FMHA-v2 Context/ViT kernels. On SM100,
SM101, and SM110 it also contains these optimized multi-head attention kernels
compiled ahead-of-time from CuTe DSL Python source. Kernel artifacts (static
library + headers) are generated locally by `kernelSrcs/build_cutedsl.py`. CMake simply links those local artifacts — no
Python, CUTLASS DSL, CuPy, or Blackwell GPU is needed at CMake build time.

> **Dependencies, `build_cutedsl.py` options, CMake integration, and
> cross-compiling/runtime deployment are shared across all CuTe DSL groups and
> documented in [`kernelSrcs/README.md`](../README.md).** This document covers
> only FMHA-specific details.

## Supported Hardware

| GPU | SM | Status |
|---|---|---|
| Blackwell datacenter (B200, GB200) | SM100 / SM103 | Primary target |
| NVIDIA Thor | SM110 | Cross-compile target (aarch64) |

## Kernel Variants

The build produces AOT-compiled kernel objects (`.o` + `.h` pairs):

| Variant | Head Dim | SWA | Mode | Causal |
|---|---|---|---|---|
| `fmha_d64` | 64 | No | LLM | Yes |
| `fmha_d128` | 128 | No | LLM | Yes |
| `fmha_d256` | 256 | No | LLM | Yes |
| `fmha_d64_sw` | 64 | Yes | LLM | Yes |
| `fmha_d128_sw` | 128 | Yes | LLM | Yes |
| `fmha_d256_sw` | 256 | Yes | LLM | Yes |
| `fmha_d64_fp8` | 64 | No | LLM (FP8) | Yes |
| `fmha_d128_fp8` | 128 | No | LLM (FP8) | Yes |
| `fmha_d256_fp8` | 256 | No | LLM (FP8) | Yes |
| `fmha_d64_sw_fp8` | 64 | Yes | LLM (FP8) | Yes |
| `fmha_d128_sw_fp8` | 128 | Yes | LLM (FP8) | Yes |
| `fmha_d256_sw_fp8` | 256 | Yes | LLM (FP8) | Yes |
| `fmha_d64_paged` | 64 | No | LLM (paged) | Yes |
| `fmha_d128_paged` | 128 | No | LLM (paged) | Yes |
| `fmha_d256_paged` | 256 | No | LLM (paged) | Yes |
| `fmha_d512_paged` | 512 | No | LLM (paged) | Yes |
| `fmha_d512_paged_bidirectional` | 512 | Runtime | LLM (paged+bidirectional block) | Yes |
| `fmha_d64_sw_paged` | 64 | Yes | LLM (paged) | Yes |
| `fmha_d128_sw_paged` | 128 | Yes | LLM (paged) | Yes |
| `fmha_d256_sw_paged` | 256 | Yes | LLM (paged) | Yes |
| `fmha_d512_sw_paged` | 512 | Yes | LLM (paged) | Yes |
| `fmha_d64_paged_fp8` | 64 | No | LLM (paged+FP8) | Yes |
| `fmha_d128_paged_fp8` | 128 | No | LLM (paged+FP8) | Yes |
| `fmha_d256_paged_fp8` | 256 | No | LLM (paged+FP8) | Yes |
| `fmha_d512_paged_fp8` | 512 | No | LLM (paged+FP8) | Yes |
| `fmha_d64_sw_paged_fp8` | 64 | Yes | LLM (paged+FP8) | Yes |
| `fmha_d128_sw_paged_fp8` | 128 | Yes | LLM (paged+FP8) | Yes |
| `fmha_d256_sw_paged_fp8` | 256 | Yes | LLM (paged+FP8) | Yes |
| `fmha_d512_sw_paged_fp8` | 512 | Yes | LLM (paged+FP8) | Yes |
| `vit_fmha_d64` | 64 | No | ViT | No |
| `vit_fmha_d72` | 72 | No | ViT | No |
| `vit_fmha_d80` | 80 | No | ViT | No |
| `vit_fmha_d128` | 128 | No | ViT | No |

The D256 variants use a dedicated TMEM and pipeline layout selected before
`cute.compile`; D256 ViT is not supported.

**LLM variants** use a fused KV cache layout `[B, 2, H_kv, S_k, D]` with causal
masking and bottom-right alignment (`WINDOW_MASK_INFERENCE`).

The D512 `BIDIRECTIONAL` mask variant unions the base causal/sliding mask with
one inclusive block interval per query row. It is compiled with runtime sliding
window support, so one AOT artifact handles both sliding layers and global
layers (the latter pass the no-limit sentinel).

**ViT variants** use packed variable-length separate Q/K/V tensors
`[total_S, H, D]` with `cu_seqlens` for ragged batching, bidirectional attention.

## Artifact Development

If you modify this kernel or its registry entries, manually regenerate the
`fmha` group before running CMake. Otherwise, CMake uses the matching prebuilt
tarball by default. Follow the shared
[CuTe DSL kernel development workflow](../README.md#cute-dsl-kernel-development-workflow)
for the supported Docker and local-venv commands, dependency versions,
cross-compilation, artifact layout, and CMake configuration.

CMake defines `CUTE_DSL_FMHA_BLACKWELL_ENABLED` only when the artifact
carries these optimized variants.

## Standalone Test / Export

```bash
cd kernelSrcs/fmha_cutedsl_blackwell

# LLM d128, no sliding window
python3 fmha.py \
  --q_shape 1,1024,14,128 --k_shape 1,1024,1,128 \
  --is_causal --is_persistent --bottom_right_align \
  --export_only --output_dir ./out --file_name fmha_d128 --function_prefix fmha_d128

# LLM d64, with sliding window
python3 fmha.py \
  --q_shape 1,1024,14,64 --k_shape 1,1024,1,64 \
  --is_causal --is_persistent --bottom_right_align \
  --window_size 4096,-1 \
  --export_only --output_dir ./out --file_name fmha_d64_sw --function_prefix fmha_d64_sw

# LLM d512 full-causal paged prefill
python3 fmha.py \
  --q_shape 1,1024,8,512 --k_shape 1,1024,1,512 \
  --is_causal --is_persistent --bottom_right_align --paged_kv \
  --export_only --output_dir ./out \
  --file_name fmha_d512_paged --function_prefix fmha_d512_paged

# LLM d512 paged prefill with a runtime causal/sliding + bidirectional-block mask
python3 fmha.py \
  --q_shape 1,1024,8,512 --k_shape 1,1024,1,512 \
  --is_causal --is_persistent --bottom_right_align --paged_kv \
  --window_size 4096,-1 --bidirectional \
  --export_only --output_dir ./out \
  --file_name fmha_d512_paged_bidirectional \
  --function_prefix fmha_d512_paged_bidirectional

# ViT d64
python3 fmha.py \
  --q_shape 1,1024,14,64 --k_shape 1,1024,14,64 \
  --is_persistent --vit_mode \
  --export_only --output_dir ./out --file_name vit_fmha_d64 --function_prefix vit_fmha_d64
```

Each invocation produces `<file_name>.h` and `<file_name>.o` in `--output_dir`.

To run reference accuracy checks (without `--export_only`):

```bash
# LLM accuracy reference
python3 fmha.py \
  --q_shape 1,8,8,128 --k_shape 1,64,8,128 \
  --is_causal --is_persistent --bottom_right_align

# ViT accuracy reference
python3 fmha.py \
  --q_shape 1,8,8,72 --k_shape 1,8,8,72 \
  --is_persistent --vit_mode
```

For an AArch64 (Thor) host-target export, `export_fmha_aarch64.sh` wraps the
above through `kernelSrcs/cutedsl_utils/cutedsl_compile_wrapper.py` (see the shared
[cross-compile section](../README.md#cross-compiling-for-aarch64-thor-and-runtime-deployment)).

## C++ Integration

`CuteDslFMHARunner` (`cpp/kernels/contextAttentionKernels/cuteDslFMHARunner.{h,cpp}`)
provides the C++ interface:

- **Module loading**: the exact AOT variant selected for an LLM or ViT dispatch
  is loaded lazily on first use. Plugins preflight that variant before launching
  preprocessing kernels, and the runner repeats the guard before its generated
  wrapper call. Loaded modules are shared and remain resident for process
  lifetime; unused variants are never loaded.
- **Dispatch**: `canImplement(headSize, smVersion)` — returns `true` for
  SM100/101/110 and head dim 64, 128, 256, or 512.
- **LLM run**: `run(qPtr, kvPtr, oPtr, cuKVSeqLens, stream, slidingWindowSize)`
  — dispatches to the appropriate d64/d128/d256 + SWA/non-SWA variant.
- **Paged LLM run**:
  `runPaged(qPtr, pagedKVPoolPtr, pageTable, oPtr, paddedCuKVSeqLens, ...)`
  — dispatches D64/D128/D256 and FP16/FP8 D512 variants through the common
  paged ABI. The FP16 D512 vision variants add `[B, S_q]` `blockBegin` and
  `blockEnd` tensors: text/padding rows contain `-1/-1`, while every row in a
  disjoint contiguous vision run repeats that run's inclusive bounds.
- **ViT run**: `run(qPtr, kPtr, vPtr, oPtr, cuSeqLens, totalSeqLen, maxSeqLen, batchSize, stream)`
  — dispatches to the appropriate d64/d72/d80/d128 variant.

Plugin (`cpp/plugins/attentionPlugin/attentionPlugin.cpp`): uses CuTe DSL FMHA
as the primary path on Blackwell, with automatic fallback to FMHA_v2.

### Sliding Window Attention

- Plugin attribute `sliding_window_size`: `-1` means disabled (default).
- At the C++ runtime boundary, `-1` is converted to `INT_MAX`.
- Runner dispatches to `_sw` variants when `slidingWindowSize < INT_MAX`.
- `window_size_right` is always `0` (causal-only), baked as a compile-time
  constant.
- `bottom_right_align` is always enabled, producing correct masking for both
  normal prefill and chunked prefill.

## Origin

`fmha.py` is derived from the CUTLASS example at
`examples/python/CuTeDSL/blackwell/fmha.py`, and `fmha_helpers.py` from
`examples/python/CuTeDSL/helpers/fmha_helpers.py`, both taken from CUTLASS commit
[`b9847690c5838ac3d909ebc163ed16c388802485`](https://github.com/NVIDIA/cutlass/commit/b9847690c5838ac3d909ebc163ed16c388802485).

Key adaptations from upstream:
- Replaced PyTorch with CuPy/NumPy
- Fused KV cache layout `(B, 2, H_kv, S_k, D)` instead of separate K/V
- Direct paged-KV loading, including four K stages and the CTA-owned two V
  stages for D512
- Dynamic batch/seq_len/nheads as runtime arguments
- Sliding window attention with compile-time dispatch
- ViT mode with packed varlen bidirectional attention
- AOT export via `export_to_c()`

## Skip-Softmax (BLASST) Threshold Calibration

The kernel implements BLASST skip-softmax ([arXiv:2512.12087](https://arxiv.org/abs/2512.12087)):
a KV tile whose local row max falls below the running max by more than
`ln(lambda)` is skipped whole (exp / row-sum / P*V elided). The feature has two
independent knobs:

- **compile-time enable** — constructing the kernel with a non-None
  `skip_softmax_threshold` compiles the skip path in (the `*_skipsoftmax`
  variants in `build_cutedsl.py` pass a sentinel `1.0` for exactly this);
  `None` compiles it out, bit-identical to the dense build.
- **runtime scale factor** — deployment never bakes a lambda: the calibrated
  **scale factor S** is an `AttentionPlugin` attribute
  (`skip_softmax_scale_factor`, set at ONNX export), passed through to the
  kernel, which derives `log2(lambda_i) = log2(S) - log2(seqlen_kv_i)` PER
  SEQUENCE (same formula as TensorRT-LLM's `threshold_scale_factor`, no
  clamp). `S = 0` (default) dispatches the dense
  kernel.

> **DEPLOYMENT RULE — long-context only.** A sequence shorter than S runs at
> `lambda > 1`, outside the calibrated domain, and degrades sharply: measured
> MMLU **-10.5pts** (edge-llm, Qwen3.5-0.8B, S=8298) and **-16.3pts**
> (official TensorRT-LLM 1.2.1, Qwen3-1.7B, S=8192, lambda ~ 10) while
> 13k-context retrieval at `lambda = 0.6` stays intact. TensorRT-LLM ships the
> identical formula with no guard — this is an ecosystem-wide property of the
> S/L semantics, not an implementation defect. **If deployment traffic
> contains requests with `L < S`, serve them with a dense engine (S = 0), or
> pick S no larger than the shortest expected request.** Accuracy gates are
> long-context only (RULER + LongBench); MMLU is reported as an advisory
> short-sequence column.

Restricted to plain causal FP16 attention (no sliding window, no FP8 input,
no ViT/bidirectional), prefill/context path only.

`calibrate_skip_softmax.py` covers the full lifecycle with two subcommands and
staged, verbose output:

```
calibrate (default) ── ModelOpt official calibration ──▶ a, b, scale factor S
      │                                                        │
      │                          re-export ONNX with --skip-softmax-scale-factor S,
      │                          rebuild engine (llm_build) — kernels untouched
      ▼                                                        ▼
evaluate ── RULER + MMLU accuracy of the deployed engine ──▶ PASS/FAIL + recommendation
```

### `calibrate` — scale factor via ModelOpt (official)

A fixed lambda yields wildly different sparsity across context lengths, so the
threshold follows `lambda = scale_factor / L` with a model-specific scale
factor. The subcommand wraps the official calibration in
`modelopt.torch.sparsity.attention_sparsity` (the same machinery behind
TensorRT-LLM's `threshold_scale_factor`): ModelOpt auto-generates a RULER
calibration set, runs one forward pass evaluating 20 built-in threshold
trials at once, and fits `scale_factor = a * exp(b * sparsity)` with scipy.
Defaults are aligned with ModelOpt's official `SKIP_SOFTMAX_CALIB` preset —
the same parameters TRT-LLM ships with: 64 samples, max_seqlen 16384
(power-of-2 length bins), chunked prefill 4096. The a/b fit does not
extrapolate beyond its calibration max_seqlen, so keep it at or above the
deployment context (short calibrations leave the sparsity targets in the
extrapolated zone — visible as a low R^2 and a narrow observed-sparsity
range in the report). Requires `torch`, `transformers`, `nvidia-modelopt`,
`scipy`, `nltk`, `wonderwords`; the model loads with
`attn_implementation="eager"`.

```bash
python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py calibrate \
    --model-dir /path/to/Qwen3-1.7B \
    --target-sparsity 0.3 0.5 --max-context 16384 \
    --cache-dir /path/with/room/modelopt-cache   # RULER gen cache; ModelOpt
                                                 # defaults to ~/.cache (quota!)
# [calibrate 1/3] load model ... [calibrate 2/3] ModelOpt calibration
#   (library output is dim and '│'-indented, this tool's lines are plain)
# [calibrate 3/3] fitted parameters and deployment scale factors
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ a = 115.037   b = 4.6992   R^2 = 0.733   (278 points)                ┃
# ┃ deployable scale factor S (lambda = S / context_length at runtime):  ┃
# ┃   target  30%  S = 471.07   (... export --skip-softmax-scale-factor) ┃
# ┃   target  50%  S = 1205.8                                            ┃
# ┃ illustration — lambda at L=4096:  0.115008 / 0.294372                ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

ModelOpt's sparsity is a simulated, all-layers-pooled metric — the deployed
kernel's per-layer skip ratio at the same lambda can differ substantially.
Treat the calibrated S as the ecosystem-consistent starting point and let
`evaluate` arbitrate which target actually deploys.

### Deploying a candidate scale factor

The kernel artifacts are lambda-free and built once. Deploying S touches only
the model artifacts:

```bash
# S becomes an AttentionPlugin node attribute in the ONNX
python -m tensorrt_edgellm.scripts.export <quantized_ckpt> <onnx_out> \
    --skip-softmax-scale-factor <S>
# rebuild the engine from it
build/examples/llm/llm_build --onnxDir <onnx_out>/llm --engineDir <engine_dir> ...
```

(Alternatively persist S as a `skip_softmax_scale_factor` key in the
checkpoint's `config.json` llm dict — the exporter reads it; the CLI flag
overrides.)

### `evaluate` — RULER + MMLU accuracy verdict for the deployed engine

The paper's accuracy instrument is RULER (its ~50%-sparsity safe-zone
conclusions come from it; retrieval-style tasks degrade first). The subcommand
samples real RULER items (HF `simonjegou/ruler`, tokenizer-filtered to the
engine's max input length), runs the deployed engine greedily, and scores by
exact-answer matching per task. With `--mmlu-samples N` it additionally scores
an N-question MMLU subset through the repo's own accuracy tooling
(`examples/accuracy`: `prepare_dataset.py` prompts + `calculate_correctness.py`
CI letter-extraction rules). Given a baseline, both deltas are gated
independently and the final verdict is **RULER AND MMLU** (exit code follows,
so it can gate CI). Pair with `llm_bench --mode prefill` for TTFT.

```bash
# 1) dense baseline: save its scores (RULER + MMLU)
python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py evaluate \
    --model-dir /path/to/Qwen3-1.7B \
    --engine-dir engines/qwen3-1.7b --llm-inference build/examples/llm/llm_inference \
    --max-context 4096 --mmlu-samples 200 --save-results dense.json

# 2) each deployed S: compare, get the verdict
python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py evaluate \
    --model-dir /path/to/Qwen3-1.7B \
    --engine-dir engines/qwen3-1.7b --llm-inference build/examples/llm/llm_inference \
    --max-context 4096 --mmlu-samples 200 --baseline dense.json --label "S=471"
# ...per-task score table with baseline/delta columns...
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ VERDICT: PASS [S=471]  (RULER AND MMLU)                   ┃
# ┃ RULER:  PASS   0.7685 -> 0.7653   drop +0.0032 (gate 0.03)┃
# ┃ MMLU:   PASS   0.6150 -> 0.6100   drop +0.0050 (gate 0.03)┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# RECOMMENDATION: this build [S=471] is validated for deployment. ...
```

### Reference results (Qwen3-1.7B NVFP4)

Perf MUST be measured with real text: the skip predicate barely fires on
random tensor content or synthetic token ids (`llm_bench` inputs), which
understates the gain to ~0. Real-text prefill TTFT (32k-capacity engines,
`lambda_eff = S / kvCacheCapacity`; sparse=x denotes the target tile-skip
ratio; dense head/tail drift <0.4%):

| Context | dense | sparse=0.3 (S=68) | sparse=0.5 (S=170) | sparse=1.0 (S=30000, ceiling) |
|---|---|---|---|---|
| <=8k (Thor) | — | within noise | within noise | within noise |
| 16k (Thor) | 459.6 ms | -0.4% | -1.5% | -13.1% |
| 24k (Thor) | 865.2 ms | -1.8% | -4.3% | -16.6% |
| 32k (Thor) | 1427.5 ms | **-5.4%** | **-8.1%** | -21.1% |
| 16k (B200) | 53.7 ms | — | **-2.4%** | — |

Accuracy at the same operating points (platform-independent): RULER@16k and
MMLU (n=1000) both PASS the 0.03 gate for sparse 0.3/0.5; a 20-needle NIAH
probe at 32k scores 16-17/20 vs dense 17/20. sparse=1.0 collapses NIAH to
9/20 — it is a perf upper-bound marker, not a deployable point.

Deployment guidance: real-text prefill gain GROWS with the skip ratio at long
context, so deploy the LARGEST S that passes the accuracy gates. The 4k
ModelOpt fit does not extrapolate — for long-context engines pick
`S = lambda_target * kvCacheCapacity` with `lambda_target` in [0.002, 0.005]
(or recalibrate with long samples). Below ~8k context the feature is
accuracy-neutral and perf-neutral; `S = 0` remains the default.

## File Map

| File | Description |
|---|---|
| `kernelSrcs/fmha_cutedsl_blackwell/fmha.py` | CuTe DSL kernel source (LLM + ViT variants) |
| `kernelSrcs/fmha_cutedsl_blackwell/fmha_helpers.py` | Helper utilities from CUTLASS |
| `kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py` | Skip-softmax threshold scale-factor calibration tool |
| `kernelSrcs/fmha_cutedsl_blackwell/fmha.patch` | Diff against upstream CUTLASS example |
| `kernelSrcs/fmha_cutedsl_blackwell/fp8_prescale.patch` | FP8 pre-scaling patch (future) |
| `kernelSrcs/fmha_cutedsl_blackwell/export_fmha_aarch64.sh` | AArch64 host-target standalone export using the CuTe DSL compile wrapper |
| `kernelSrcs/build_cutedsl.py` | Unified pre-build script: compiles all CuTe DSL kernel groups |
| `cmake/CuteDsl.cmake` | Unified CMake module: validates and links prebuilt artifacts |
| `cpp/kernels/cuteDSLArtifact/{arch}/{artifact_tag}/` | Local artifacts generated by `build_cutedsl.py` |
| `cpp/kernels/contextAttentionKernels/cuteDslFMHARunner.h` | C++ runner header |
| `cpp/kernels/contextAttentionKernels/cuteDslFMHARunner.cpp` | C++ runner implementation |
| `cpp/plugins/attentionPlugin/attentionPlugin.cpp` | TRT plugin integration |
| `cpp/kernels/posEncoding/applyRopeWriteKV.cu` | RoPE kernel for CuTe DSL KV layout |
