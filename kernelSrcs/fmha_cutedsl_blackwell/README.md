# CuTe DSL FMHA Kernels (Blackwell SM100/SM101/SM103/SM110)

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

AOT builds receive `--target_sm` from `build_cutedsl.py`, using the same
CLI-based architecture selection as FMHA decode. Direct Python runs should
also pass the flag to enable the hardware reductions.

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

**Hardware row-max loads:** On SM110 (legacy SM101), D64 FP8 ViT and full-causal LLM
kernels use `LDTM.STAT` to load scores and compute partial row maxima on unmasked KV tiles.
Masked tiles use software reduction after applying their masks; STAT-enabled kernels discard
the hardware maxima on these tiles. FP16, sliding-window, and other head-dimension
variants also retain software reduction. SM100 and SM103 remain supported with
software row-max reduction; this optimization is enabled only for SM110/legacy SM101.

**MMA row sums:** D64 FP8 kernels with hardware row-max loads and without BLASST
compute normalization sums with a second MMA, multiplying the stored probabilities
by a shared-memory matrix of ones. The result occupies unused TMEM columns after
each output tile and uses the existing MMA-to-correction synchronization.
These sums use probabilities rounded to the PV input dtype, so outputs can differ
from the software sum of unrounded FP32 probabilities. Other variants retain
software row sums. Register budgets are unchanged.

Learned sinks use the unrounded FP32 sum for their attention-mass factor while
output normalization retains the MMA sum. Sink-free paged launches use a
separate GPU specialization to avoid the additional reduction work.

The d64–d128 kernels with 16-bit inputs skip the identity O rescale: the
correction warpgroup takes a warp vote and skips the O rescale (a 128-row TMEM
read-modify-write per KV tile) when every lane's scale is exactly 1.0. With
skip-correction on, the carried row max makes that the common case, and
multiplying by 1.0 is the identity, so the output is bit-identical. There is no
switch for it; the kernel decides at compile time from the input dtype — the FP8
variants keep the unconditional rescale because their smaller skip-correction
threshold moves the carried max often, so the identity case is rare and the vote
would only add overhead.

`vit_fmha_d72` is built with the 5-deep K/V ring (`--kv_stage 5`) and `vit_fmha_d80`
with the 4-deep one (`--kv_stage 4`); d64, d96 and d128 keep the default depth of 3.
The ring depth only shows on long sequences: at 3136 tokens every FP16 variant is
within ±1 % across all depths that fit the 227 KB smem budget (d72/d80 up to 7 stages,
d64 up to 10, d96 up to 5); at 7056 tokens d72 gains 7 % from 3 to 4 stages and is flat
from 4 to 7 (5 is the measured optimum at both lengths), d80 gains 5 % from 3 to 4 and
is flat beyond, d64 is flat at every depth, and d96 gets slower with every added stage
(+8 % at 4 and at 5 — its 24.5 KB slots push the CTA to the smem limit), so it stays
at 3. The depth is baked into the artifact; changing it is a rebuild plus revalidation.

**d72 sector alignment** (automatic for a head size whose compact pitch is not a
32 B multiple; a no-op for the others). The padded K=80 tile is fetched as five 32 B
TMA boxes per row. With the compact 144 B head pitch the token stride (H × 144 B) is a
32 B multiple, but every odd head starts 16 B into an L2 sector, so each of its boxes
straddles two sectors and the MMA warp waits on K/V. The kernel resolves that parity
per work unit from the head index and issues the Q/K/V boxes of the misaligned heads
from D coordinate −8: those columns are out of bounds for the descriptor (extent 72),
so TMA zero-fills them without a fetch and every box row lands on a sector boundary.
The tile then holds `[0 × 8 | 72 valid]` instead of `[72 valid | 0 × 8]`; Q·Kᵀ is
mathematically identical (Q and K of one head share the shift and the zero columns
contribute nothing — the 72 valid columns fall into different K=16 MMA phases, so the
fp32 partial sums are grouped differently and S of a shifted head may differ from the
unshifted layout in the last ulp; only the identity-rescale skip above is bit-identical),
the O rescale covers the whole tile, and the correction epilogue reads O from TMEM
column 8 for those heads. That read covers 8 columns past the 80-column accumulator,
into the unused gap before the next O slot (checked at compile time); they land in the
smem columns 72..79 that the TMA store drops. Buffers `[total_S, H, 72]`, TMA
descriptors, smem layouts, the O store and the AOT wrapper are unchanged, so nothing
outside the kernel knows about it. The O rows of the shifted heads are therefore still
written 16 B into a sector (each 32 B store box row touches two sectors); this is
deliberate — O is a few percent of a work unit's K/V traffic and the bulk store is off
the critical path. Shifting the O store coordinate instead of the TMEM read would align
the writes too, at the cost of teaching the partial-tile store about the shift; that is
the follow-up if the write side ever shows in a profile. Correctness never depends on
the parity guess (a wrongly shifted head is only slower); the fast path assumes
32 B-aligned Q/K/V base pointers and an even head count.

Correction-path changes must be validated with `--vit_ramp_k
--no-enable_skip_correction` (K grows along the sequence, so the row max moves
on most KV tiles and the rescale is a real multiply); random data leaves the
rescale at 1.0 and cannot exercise it.

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

# ViT at the shipped shape and the adversarial correction-path check (the rescale
# really multiplies; random data leaves it at the identity and skips it)
python3 fmha.py --q_shape 1,3136,16,80 --k_shape 1,3136,16,80 --is_persistent --vit_mode
python3 fmha.py --q_shape 1,3136,16,80 --k_shape 1,3136,16,80 --is_persistent --vit_mode \
  --vit_ramp_k --no-enable_skip_correction

# d72 at the shipped configuration (5-deep K/V ring; the sector alignment is automatic)
python3 fmha.py --q_shape 1,3136,16,72 --k_shape 1,3136,16,72 --is_persistent --vit_mode \
  --kv_stage 5
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
  SM100/101/103/110 and head dim 64, 128, 256, or 512.
- **LLM run**: `run(qPtr, kvPtr, oPtr, cuKVSeqLens, stream, slidingWindowSize)`
  — dispatches to the appropriate d64/d128/d256 + SWA/non-SWA variant.
- **Paged LLM run**:
  `runPaged(qPtr, pagedKVPoolPtr, pageTable, oPtr, paddedCuKVSeqLens, ...)`
  — dispatches D64/D128/D256 and FP16/FP8 D512 variants through the common
  paged ABI. The FP16 D512 vision variants add `[B, S_q]` `blockBegin` and
  `blockEnd` tensors: text/padding rows contain `-1/-2`, while every row in a
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
- **runtime scale table** — deployment never bakes a lambda: the calibrated
  **scale factor S** travels in the engine `config.json` as
  `skip_softmax_prefill_scale_table`, rows of `[seqlen_bin, S]` (a flat S is the one-row
  table `[[0, S]]`). The runtime uploads it once to the `skip_softmax_scale`
  plugin input; per sequence the kernel floor-looks-up S by its **actual** KV
  length (a second `cum_seqlen_k` argument — the causal path's primary one is
  padded to the batch's longest query) and derives
  `log2(lambda_i) = log2(S) - log2(seqlen_kv_i)`. A sequence below the
  first bin, or with `S >= seqlen_kv` (`lambda >= 1` would pass every tile),
  skips nothing — the dense result to fp16 rounding, inside the skip cubin. The
  `skip_softmax_prefill_input` AttentionPlugin attribute (set at export) only
  marks the layers that carry the table input.
- **dense dispatch** — the runtime binds the table at length 0 when the whole
  prefill batch sits below the table's first skippable length, so short prompts
  run the independently compiled dense cubin with zero skip-kernel tax.

Restricted to plain causal FP16 attention (no sliding window, no FP8 input,
no ViT/bidirectional), prefill/context path only.

`calibrate_skip_softmax.py` covers the full lifecycle with three subcommands and
staged, verbose output:

```
calibrate (default) ── ModelOpt official calibration ──▶ a, b, per-length S tables
      │              (no calibrate --target-sparsity: tiers 50/60/75/80 %, 75 % deploys)
      │                                                        │
      ▼                                                        ▼
price ── export --prefill-target-sparsity R + llm_build per tier ──▶ TTFT speedup, RULER,
      │  (runs automatically after a ladder calibration)      LongBench per length
      ▼
evaluate ── RULER + MMLU accuracy of one deployed engine ──▶ PASS/FAIL + recommendation
```

The one-command form: `calibrate --model-dir <HF ckpt> --max-seqlen 40960
--bin-step 2048` calibrates every tier's table, records them in the checkpoint
`config.json` (`skip_softmax_prefill_scale_tables`, the deploying one also as
`skip_softmax_prefill_scale_table`), then prices the ladder at 8k / 16k / 32k / 40k.
`tensorrt-edgellm-export` and `tensorrt-edgellm-build` need no skip flag
afterwards; `export --prefill-target-sparsity R` switches to another tier's table.

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
# ┃   target  30%  S = 471.07   (export --prefill-target-sparsity 0.3)   ┃
# ┃   target  50%  S = 1205.8                                            ┃
# ┃ illustration — lambda at L=4096:  0.115008 / 0.294372                ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

ModelOpt's sparsity is a simulated, all-layers-pooled metric — the deployed
kernel's per-layer skip ratio at the same lambda can differ substantially.
Treat the calibrated S as the ecosystem-consistent starting point and let
`evaluate` arbitrate which target actually deploys.

#### Shape advisor — mass priced against sparsity

ModelOpt's calibration counts the tiles each trial lambda skips (sparsity,
which predicts speed). The tool hooks that call
(`FlashSkipSoftmax.calc_correction_factor_and_p`, no ModelOpt change) and, in
the same pass, adds the exact softmax **mass** those tiles carry (which predicts
accuracy):

```
s(λ) = |{tiles with exp(m̃ − m_run) < λ}| / N_tiles           ModelOpt: tile COUNT
m(λ) = Σ_skipped mass(tile) / Σ_all mass(tile)                added:    tile MASS
```

Both are read at the lambda the kernel deploys per request (`S / L` from the
deployed table, dense requests excluded; when no tier is reachable, at each
request's own most sparse trial) and pooled per request; the mass is read at
its tail over requests (max), not the mean. One line, `--mass-ratio` (0.3):

- `m / s >= 0.3` → **off**: the skipped tiles carry their count share — flat
  score distribution (the few full-attention layers of a hybrid). Every tile
  is paid at full price; forcing a tier costs points for ~1.0x.
- `m / s < 0.3` → **on**: the skipped tiles are near-zero (A-shaped scores).

The report lists sparsity, mass share (tail / mean), their ratio, the verdict,
the same per length bin, and the layers whose own ratio is flat; it is recorded
as `skip_softmax_prefill_shape_advice` in the checkpoint `config.json` (informational —
export/build ignore it). `--no-shape-advisor` skips the measurement (one extra
row-tile softmax per calibration call).

### Deploying a candidate scale factor

The kernel artifacts are lambda-free and built once. Deploying S touches only
the model artifacts. The exporter has exactly four skip-softmax flags, two per
phase: `--skip-softmax-{prefill,decode}-scale-table <json|none>` and
`--{prefill,decode}-target-sparsity R`. A table flag outranks every checkpoint
key of its phase; the literal `none` (or a file holding `[]`) makes that phase
dense; without a flag the checkpoint's `skip_softmax_<phase>_*` keys decide.

```bash
# a flat S is the one-row table [[0, S]]; written to the exported config.json as
# skip_softmax_prefill_scale_table
echo '[[0, <S>]]' > prefill_table.json
python -m tensorrt_edgellm.onnx.cli <quantized_ckpt> <onnx_out> \
    --skip-softmax-prefill-scale-table prefill_table.json
# rebuild the engine from it
build/bin/llm_build --onnxDir <onnx_out>/llm --engineDir <engine_dir> ...
```

(A calibrated per-length table goes in the same way, a JSON list of
`[seqlen_bin, S]` rows; `--prefill-target-sparsity R` picks tier R's table from a
calibrated checkpoint instead.) Re-targeting a deployed engine needs no re-export:
edit `skip_softmax_prefill_scale_table` in the engine `config.json`.

### `price` — speedup and accuracy drop of every tier, per prompt length

Runs after a ladder calibration (suppress with `--no-price`) or standalone on a
calibrated checkpoint. For the dense reference (`--skip-softmax-prefill-scale-table none
--skip-softmax-decode-scale-table none`) and every tier (`export --prefill-target-sparsity R`, which picks that tier's table) it
exports, builds one engine sized to the longest `--lengths` entry, and measures
per length:

- **TTFT** — by default (`--ttft-source real`) `llm_inference --dumpProfile` on
  four cnn_dailymail prompts tokenized to the length (cached under
  `--work-dir/prompts/`; distinct article starts defeat prefix caching); the skip
  predicate barely fires on synthetic tokens, so `--ttft-source bench`
  (`llm_bench --mode prefill`) is kept only as the cheap dense-speed check.
  Median of `--bench-repeats` runs.
- **attention-kernel speedup and share** — when `nsys` is on PATH
  (`--no-kernel-profile` to skip) one profiled run per (engine, length) sums the
  kernels matching `--attention-kernel-regex` (both FMHA families by default):
  `kernel_speedup` = dense / tier attention time, `attn_share` = attention /
  all-kernel time of the dense engine. The two numbers bound what end-to-end can
  gain: `TTFT speedup <= 1 / (1 - share + share / kernel_speedup)`.
- **RULER** via `evaluate` (only at the dataset's 4k / 8k / 16k configurations) and
  **LongBench** via `longbench_eval.py`, on `--accuracy-lengths` (default: every
  length; restrict it on slow devices — LongBench at 32k is the expensive part).

Binaries default to `build/bin/{llm_build,llm_bench,llm_inference}`;
when they are missing the step is skipped with a hint. Exports, engines,
prompts, nsys reports and finished RULER/LongBench scores land in
`--work-dir` (default `<model-dir>-skip-softmax-price`) and are reused on a
re-run. Results go to the
checkpoint `config.json` (`skip_softmax_price_list`), to `--work-dir/price.json`,
and to an HTML report (`--report`, default `--work-dir/skip_softmax_report.html`).

```bash
python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py price \
    --model-dir /path/to/Qwen3-8B --lengths 8192 16384 32768 40960
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ skip-softmax price list (vs dense; accuracy delta in points, RULER @4k/8k/16k) ┃
# ┃   tier         8192            16384            32768            40960         ┃
# ┃   50%   1.02x R-0.003 LB+0.001  1.05x R-0.005 LB-0.002  1.09x LB-0.004  1.11x …┃
# ┃   75%   1.04x R-0.008 LB-0.004  1.10x R-0.012 LB-0.006  1.17x LB-0.009  1.21x …┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# recorded skip_softmax_price_list in /path/to/Qwen3-8B/config.json
```

(Values above illustrate the layout only; `k` is the attention-kernel speedup.)
The same dictionary is stored under `skip_softmax_price_list` in the checkpoint
`config.json` for later inspection.

### `report` — one figure per model: kernel and TTFT speedup against length

```bash
python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py report \
    --price-json work/qwen3-8b/price.json work/llama-3.1-8b/price.json \
    --out skip_softmax_report.html
```

Renders the price runs into a self-contained HTML page: per model one row with
the attention-kernel speedup (left) and the end-to-end TTFT speedup (right)
over prompt length on a log-2 axis, one line per tier, the deploy tier's peak
annotated, and below it the table of TTFT, kernel speedup and RULER/LongBench
drop per length. `price` writes the same page for its single run; `report`
merges runs from several models or platforms (`--price-json` files or priced
checkpoints via `--model-dir`).

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
    --engine-dir engines/qwen3-1.7b --llm-inference build/bin/llm_inference \
    --max-context 4096 --mmlu-samples 200 --save-results dense.json

# 2) each deployed S: compare, get the verdict
python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py evaluate \
    --model-dir /path/to/Qwen3-1.7B \
    --engine-dir engines/qwen3-1.7b --llm-inference build/bin/llm_inference \
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
