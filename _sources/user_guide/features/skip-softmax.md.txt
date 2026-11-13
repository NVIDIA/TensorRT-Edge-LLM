# Skip-Softmax Attention

## Overview

Skip-softmax attention reduces the cost of **long-context attention** in prefill and in
decode by letting the attention kernel omit work whose result is known to be negligible.
FlashAttention streams the KV sequence tile by tile and tracks the largest score seen so
far; a tile whose scores all lie far below that maximum ends up with softmax weights
below a calibrated threshold $\lambda$ regardless of the rest of the row, so the kernel
does not finish it. The rule is the BLASST criterion
([arXiv 2512.12087](https://arxiv.org/abs/2512.12087)): nothing is retrained, no weight
is modified, and the KV cache stays exact.

[Section 1](#skip-softmax-background) explains the criterion, [section 2](#skip-softmax-problems)
the two changes made to deploy it on an edge engine, [sections 3 and 4](#skip-softmax-performance)
the measured accuracy and speed, and [section 5](#skip-softmax-usage) the workflow.

**Key Points**:
- Kernel-only: the BLASST test runs inside the attention kernels -- no retraining,
  no re-quantization, no graph change; the kernel artifacts are threshold-free and built once
- Both phases skip real work: the prefill kernel drops a tile's softmax and $P \cdot V$,
  the paged decode kernel drops the load of the tile's $V$ page (see
  [Scope](#skip-softmax-scope) for the kernel families that ship today)
- Per-length scale table with a built-in guard: every sequence in a batch uses the
  threshold calibrated for its own length; below the first calibrated length the
  independently compiled dense kernel runs at zero overhead
- Prefill and decode are independent: one table and one `config.json` key per phase;
  a deployed engine is re-targeted, or either phase set dense, by editing that key
- Off by default; enabled only when the checkpoint or export carries a calibration

---

(skip-softmax-background)=
## 1. How skip-softmax works

**What can be skipped.** Any tile whose scores all lie far below the row's running
maximum can be omitted; the speedup is the share of such tiles. Softmax turns a row
of scores $x_1 \ldots x_L$ into weights

$$
p_i = \frac{e^{\,x_i - x_{\max}}}{\sum_j e^{\,x_j - x_{\max}}}
$$

so what matters for each key is its **gap** to the row maximum, not its score: a gap of
20 nats is a weight of $e^{-20} \approx 2 \times 10^{-9}$, below the resolution of the
FP16 accumulator. On ordinary text most of a long row is of this kind: a few keys (the
first token and the ones just behind the query) carry nearly all the weight and the
thousands in between sit near $10^{-4}$ each -- the *A-shape* of causal attention.
Measured on Qwen3-8B, only 7.7% of far-field blocks survive a $10^{-3}$ threshold; the
rest is computed in full in both phases -- as FLOPs in prefill, where attention grows
with the square of the prompt, and as bytes in decode, where every generated token
re-reads the whole KV cache.

**The tile vote inside FlashAttention.** FlashAttention already tracks the running row
maximum $m_{\text{run}}$ as it streams KV tiles, so the gap of a whole tile is available
for free: take the tile's largest score and subtract $m_{\text{run}}$. BLASST turns that
into a vote taken once per 128 x 128 tile, before any exponential is computed:

$$
g = \max_{\text{tile}} S - m_{\text{run}}, \qquad
\text{skip the tile if } g < \ln\lambda \quad (\lambda \in (0, 1))
$$

Where the vote sits in the online-softmax loop, and what a skipped tile saves:

```text
Q = load(Q_tile)                       # TMA           HBM -> SMEM, once per CTA
m_run, l, acc = -inf, 0, 0
for kv_tile in sequence:
    K     = load(K_tile)               # TMA           HBM -> SMEM        -- always runs
    S     = Q @ K.T                    # Tensor Cores  BMM1               -- always runs
    m_new = max(m_run, rowmax(S))      # CUDA Cores    running max        -- always runs
    if rowmax(S) - m_new < ln(lambda): # the vote: one compare per tile
        continue                       # skipped: nothing below runs for this tile
    alpha = exp(m_run - m_new)         # CUDA Cores    rescale factor for the new max
    m_run = m_new
    P     = exp(S - m_new)             # CUDA Cores    exponentials (SFU)
    l     = l * alpha + rowsum(P)      # CUDA Cores    row sum, the denominator
    V     = load(V_tile)               # TMA           HBM -> SMEM
    acc   = acc * alpha + P @ V        # Tensor Cores  BMM2, the numerator
O = acc / l                            # CUDA Cores    normalize once per row
store(O)                               # TMA           SMEM -> HBM
```

A skipped tile is omitted entirely rather than computed in lower precision, and the
error is bounded by construction: every weight the tile would have contributed is below
$\lambda$ (the normalizer is at least 1, the row maximum's own term). The vote is
conservative in two ways: a tile that raises the running maximum has $g = 0$ and is never
skipped, and a single query row above the threshold keeps the whole tile for all 128
rows. Which saving dominates depends on the phase: **prefill** is compute-bound, so the
omitted arithmetic is the gain (roughly two thirds of the tile); **decode** is
bandwidth-bound, so the omitted $V$ page load is the gain, and since $K$ still streams in
full the decode ceiling is the point where only $K$ is read, about 1.5x over dense.
Every query votes on every KV tile independently; nothing is evicted and nothing carries
over between requests.

**The knob: one scale factor per checkpoint.** A fixed $\lambda$ does not give a fixed
amount of skipping. The weights of a row of $L$ keys sum to 1, so the *average* key
receives $1/L$; the $\lambda$ that skips half the tiles at 8k skips almost everything at
40k. The paper's calibration therefore expresses the threshold in units of that average
share:

$$
\lambda = \frac{S}{L}, \qquad S = \lambda \cdot L \;\text{ is held constant per checkpoint}
$$

A tile is skipped when its peak weight is below $S$ times the average key's share, which
keeps the sparsity constant as $L$ changes. $S$ is what calibration produces and what
travels in `config.json`; the kernel divides it by each sequence's own length. It is
calibrated per checkpoint ([section 5, step 1](#skip-softmax-usage)):
`calibrate_skip_softmax.py` runs ModelOpt's `attention_sparsity` calibrator over
auto-generated RULER samples -- long essays with planted facts, so the far field carries
real evidence rather than text on which skipping would cost nothing -- and records the
$S$ that reaches the target sparsity.

(skip-softmax-problems)=
## 2. From paper to deployment: two changes

An edge engine serves every request length -- 1k chat turns and 40k documents -- from a
single build. Shipping the recipe above under that constraint changed two things: **how
the threshold travels** from calibration to the kernel, and **what calibration measures**.

(skip-softmax-table)=
### 2.1 A discretized per-length table replaces the S / L formula

**Problem.** $\lambda = S / L$ is one calibrated number plus a formula, and the formula
fails at both ends of the length axis. *Below $S$* a request runs at $\lambda > 1$: every
tile satisfies the test, the softmax is effectively not computed, and accuracy collapses
(measured MMLU **−10.5 points** on ~1k-token prompts with an $S$ calibrated for long
context). *Above $S$* the best threshold is not linear in $1 / L$: with $S$ calibrated for
75% at 16k, Qwen3-8B reaches 76.6% at 8k, 72.8% at 16k, 69.6% at 32k and 66.7% at 40k --
on target only at its own anchor. Calibrating at a longer anchor over-skips every shorter
request instead (−7.0 LongBench points at 4k for an $S$ calibrated at 32k).

**Change.** Calibrate one $S$ per 2k length bin at the same target (`--bin-step 2048`),
store the rows `[seqlen_bin, S]` in `config.json`, and let the kernel look up each
sequence's $S$ by its actual KV length (floor to the bin, then $S / L$ within the bin,
where the formula holds). The runtime uploads each phase's table once to its own
attention-plugin input, `skip_softmax_prefill_scale` and `skip_softmax_decode_scale`;
the prefill kernels read the first, the decode kernel the second. Because the lookup
happens where the true per-sequence length is known, ragged batches and chunked prefill
are correct by construction; the lookup has no measurable cost (the dense and table
kernels are identical in registers, stack and local memory).

The same lookup closes the short end. The kernel forces the threshold to $-\infty$
whenever $S \ge L$ or $L$ is below the first calibrated bin, which yields exact dense
attention. A guarded skip kernel still runs the vote and carries the register allocation
of both paths -- roughly 5--10% slower than dense at short lengths -- so the engine
carries **two independently compiled cubins**. Before an initial prefill the runtime
checks whether any sequence in the batch reaches the table's first skippable length; if
none does, the plugin launches the dense cubin.

```text
# host, once per initial prefill (empty KV cache)
if max(L_i for every sequence i in the batch) < first_skippable_length(table):
    launch(dense_cubin)                 # no vote, no table: 1.000x, bit-exact
else:
    launch(skip_cubin, table)           # the kernel decides per sequence:
                                        #   S_i = table[floor(L_i)]
                                        #   L_i below the first bin or S_i >= L_i
                                        #       -> guard: skip nothing (exact dense)
                                        #   otherwise lambda_i = S_i / L_i
# with KV history (chunked prefill, multi-turn) the host does not know the total
# length: the skip cubin is launched and the in-kernel guard decides per sequence
```

**Result.** One per end of the length axis:

- **Lossless fallback to dense.** Prompts below the first bin run the separately compiled
  dense kernel and measure **1.000x** versus dense -- bit-exact, no vote, no table.
  Inside the skip kernel the per-sequence guard gives the same exact result to any slot
  whose $S \ge L$.
- **The table holds the target sparsity by discretizing it.** Above the first bin every
  length sits on the calibrated target: for a 75% target Qwen3-8B measures 73.2--74.7%
  sparsity at every length from the first bin to 40k (a 2k-period ripple inside each
  bin), where the flat $S$ calibrated at 16k drifted from 76.6% at 8k to 66.7% at 40k.
  On LongBench (Qwen3-8B, 4k--32k) the table costs 1.0 point on average versus dense and
  nothing at 4k; a flat $S$ calibrated at 8k / 16k / 32k costs 2.0 / 3.4 / 5.9 points.

(skip-softmax-advisor)=
### 2.2 Calibration weighs the skipped tiles, not just counts them

**Problem.** Running the standard recipe on hybrid models (Qwen3.5-9B: 24 gated delta-net
+ 8 full-attention layers; Gemma4-E2B: 28 sliding-window + 7 global layers) finds no
solution for the tiers above 75%: on a hybrid's few full-attention layers about 40% of
tiles are near-exact ties with the row maximum, because those layers carry every
long-range dependency and have no far field. Sparsity alone does not distinguish the two
cases. At the same 50% tier, the same 6 of 12 tiles skipped: on an A-shaped layer the
skipped tiles are far-field tiles and carry about 2% of the softmax mass; on a flat layer
every tile carries about the same, so the skipped tiles carry about 50% -- a ~25x
difference in dropped mass for the same tile count.

```text
A-shaped row (sink and local band tall, far field near zero)
tile  mass(b)
t0   ██████████████████████████████████████    95
t1   █████████                                 22
t2   ░░░                                        8   skipped
t3   ░░                                         6   skipped
t4   ░░                                         5   skipped
t5   ░░                                         4   skipped
t6   ░░                                         5   skipped
t7   ░░                                         6   skipped
t8   ████████                                  21
t9   ██████████                                24
t10  ████████████████████████████              70
t11  ███████████████████████████████████       88
6 of 12 tiles skipped -> about 2% of the row's mass dropped

flat row (a hybrid's full-attention layer: every tile about the same)
tile  mass(b)
t0   ██████████████                            36
t1   ░░░░░░░░░░░░░                             32   skipped
t2   ███████████████                           37
t3   ░░░░░░░░░░░░                              31   skipped
t4   ░░░░░░░░░░░░░                             33   skipped
t5   ██████████████                            35
t6   ░░░░░░░░░░░░                              30   skipped
t7   ██████████████                            36
t8   ░░░░░░░░░░░░░                             32   skipped
t9   ███████████████                           38
t10  ░░░░░░░░░░░░                              31   skipped
t11  ██████████████                            36
6 of 12 tiles skipped -> about 50% of the row's mass dropped
```

**Change.** ModelOpt's calibration measures exactly one quantity per trial $\lambda$,
the tile **sparsity** $s(\lambda) = |\{\,b : e^{\tilde m_b - m_{\text{run}}} < \lambda\,\}| / N_{\text{tiles}}$,
which predicts speed and is the only parameter the calibrator selects on.
`calibrate_skip_softmax.py` hooks the same call and adds the share of the softmax
numerator those tiles carry -- the **mass share**, which predicts accuracy:

$$
m(\lambda) = \frac{\sum_{b\,\in\,\text{skipped}} \text{mass}(b)}{\sum_{b} \text{mass}(b)},
\qquad
\text{mass}(b) = \sum_{i \in b} e^{\,x_i - x_{\max}}
$$

Both are read at the $\lambda$ the kernel will deploy for each request, pooled per
request, with the mass taken at its tail over requests (max, not mean). The verdict is
the ratio $r = m / s$ against `--mass-ratio` (default 0.3): $m \ll s$ means the skipped
tiles carry almost nothing -- A-shaped scores, **ON**; $m \approx s$ means they carry
their count share -- flat scores, **warn and recommend OFF**.

**Result.** The advisor's two numbers next to what deployment then measured:

| model | layers that can skip | $s$ (target) | $m$ (mass dropped) | advice | e2e @40k | LongBench @32k vs dense |
|---|---|---|---|---|---|---|
| Qwen3-8B | 36 / 36 full attention | 50% | 0.8% | ON · $m \ll s$ | ~1.1x | 0.975 (−2.5%) |
| Llama-3.1-8B | 32 / 32 full attention | 50% | 3.0% | ON · $m \ll s$ | ~1.08x | 0.968 (−3.2%) |
| Qwen3.5-9B | 8 / 32 (hybrid GDN) | 50% | 48% | OFF · $m \approx s$ | ~1.03x | 0.818 (−18.2%) |
| Gemma4-E2B | 7 / 35 (hybrid sliding) | 50% | 50% | OFF · $m \approx s$ | ~1.00x | 0.846 (−15.4%) |

The verdict is recorded in the checkpoint `config.json` as
`skip_softmax_prefill_shape_advice` and printed at the end of every calibration. It is a
recommendation, not a certification: an ON verdict still goes through the RULER /
LongBench gate before deployment.

---

(skip-softmax-performance)=
## 3. Accuracy Evaluation

Six target-sparsity tiers, each a per-length table calibrated with the workflow in
[section 5](#skip-softmax-usage), scored against the dense engine of the same checkpoint
(Qwen3-8B FP16) on RULER (six tasks, n = 8 per length) and LongBench V1 (13 English tasks,
n = 30 per task, greedy). Values are score ratios versus dense; run-to-run noise is ±0.02.

| tier | RULER @8k | @16k | @24k | @32k | LongBench @8k | @16k | @24k | @32k |
|------|-----------|------|------|------|---------------|------|------|------|
| 50% | 1.001 | 1.009 | 1.020 | 0.975 | 0.982 | 0.992 | 0.996 | 0.993 |
| 60% | 0.981 | 1.001 | 1.023 | 0.973 | 0.991 | 0.994 | 0.977 | 0.995 |
| 70% | 0.997 | 1.018 | 0.993 | 0.979 | 1.004 | 1.002 | 1.001 | 0.993 |
| 75% | 0.997 | 0.978 | 0.976 | 0.960 | 0.983 | 0.983 | 0.986 | 0.989 |
| 80% | 0.997 | 0.983 | 0.940 | 0.915 | 1.003 | 0.990 | 0.992 | 0.987 |
| 85% | 0.997 | 0.980 | 0.920 | 0.902 | 0.920 | 0.945 | 0.941 | 0.934 |

Every tier up to 75% stays inside the noise band at 8k--24k; the cost appears at 32k,
where the 80% and 85% tiers fall to 0.915 and 0.902 on RULER -- retrieval tasks degrade
first as sparsity rises. The decode table is scored the same way (decode arm against the
dense arm of the same engine, prefill table off): at the 75% decode tier on Thor,
Qwen3-8B RULER@16k 0.8815 → 0.8815 and LongBench@32k 0.3336 → 0.3335;
Llama-3.1-8B-Instruct RULER@16k 0.8814 → 0.8833 and LongBench@32k 0.3757 → 0.3762 --
neutral within noise.

Choosing a tier: fix the accuracy ratio you can accept, then take the highest tier that
stays above it at every length you serve. `price` produces this table for any calibrated
checkpoint, and `--prefill-target-sparsity` / `--decode-target-sparsity` switch tiers
without recalibrating.

---

## 4. Performance Benchmark

```{note}
The numbers in this section are rounded ranges from our own measurements with
`calibrate_skip_softmax.py price` (real-text prompts, batch 1, per-length table with the
guard, FP16 KV cache unless stated; Thor with clocks pinned by `jetson_clocks`). A Perflab (PBR) validation of the same configurations is in progress; the
exact per-model values will replace the ranges when it completes.
```

(skip-softmax-kernel-performance)=
### Kernel Performance

The speedup increases monotonically with the tier and saturates: on Qwen3-8B at 40k the
six tiers from 50% to 85% move TTFT from roughly 1.12x to 1.17x, and the last two steps
add almost nothing. Below ~8k every tier reads 1.00x (the guard runs dense).

**Prefill kernel.** At the 75% tier and 32--40k context on Thor, the prefill attention
kernel runs roughly **1.3--1.5x** faster than dense on the pure-attention decoders
(Qwen3 0.6B--14B, Qwen2.5, Llama-3.1, the text stacks of Cosmos-Reason2 and InternVL3.5)
-- less than the two-thirds-of-the-tile arithmetic suggests, because on Blackwell the
attention kernel runs one CTA per SM and a TMEM-/latency-bound floor remains once the
removable work is gone. The hybrids (Qwen3.5-9B, Gemma4-E2B) only calibrate at 40--50%
and gain roughly 1.1--1.2x at the kernel level. GEMM precision does not affect the
attention kernel (the NVFP4, FP8 and INT4-AWQ engines see the same kernel speedup); an
FP8 KV cache does -- its dense kernel is already much faster, so the skip gain on top of
it shrinks to roughly 1.1--1.2x.

**Decode kernel.** With the 75% decode table on Thor, Qwen3-8B and Llama-3.1-8B run
roughly **1.2x** faster per call at 8k and **1.4--1.5x** at 24--32k, close to the
only-$K$-is-read ceiling of ~1.5x. At 4k almost nothing is skipped and the cost of the
vote itself (a few microseconds per call) is within run-to-run noise end to end.

(skip-softmax-e2e-performance)=
### End-to-end Performance

End-to-end TTFT follows from the kernel speedup and attention's share of prefill by
Amdahl's law:

$$
\text{TTFT speedup} = \frac{1}{1 - \text{share} \cdot \left(1 - \frac{1}{\text{kernel}}\right)}
$$

**Prefill TTFT.** Across the pure-attention decoders the peak prefill TTFT speedup with
FP16 GEMMs is roughly **1.08--1.19x on Thor**; the smallest models gain most because
attention is the largest share of their prefill. Prompts of 1k--4k run the dense kernel
and measure 1.00x.

**Decode TPOT.** The same relation holds per generated token: roughly **1.07--1.08x** at
32k context on Thor (attention is ~20--25% of the step there), ~1.03--1.04x at 16k,
within noise at 8k and below.

(skip-softmax-precisions)=
Quantizing the GEMMs raises attention's share of prefill, so the same attention kernel
gives different end-to-end numbers on Qwen3-8B at 40k on Thor: roughly **1.1x** with FP16
GEMMs, **1.2x** with FP8, **1.3x** with NVFP4, and ~1.08x with INT4-AWQ. The faster the
rest of the model, the larger the skip-softmax gain; an FP8 KV cache works the other way
and leaves only a few percent end to end.

(which-models-benefit)=
Which checkpoints benefit is governed by how many layers are full attention with an
A-shaped score distribution, not by model size or head dimension. Pure-attention decoders
reach the range above; hybrids (Qwen3.5-9B, Gemma4-E2B) cannot calibrate above 50% / 40%
and the shape advisor reports OFF ([section 2.2](#skip-softmax-advisor)); hybrid MoE
gains a few percent at most (Qwen3.6-35B-A3B only above 36k, Nemotron-3.5-Lightning-30B
about 1% because a Mamba prefill kernel dominates its TTFT); fine-tuned or
reasoning-distilled variants tolerate less sparsity than their base (R1-Distill-7B passes
the gates only at 50%).

---

(skip-softmax-usage)=
## 5. Using skip-softmax

Skip-softmax is checkpoint-driven. A one-time calibration records the per-phase tables in
the checkpoint's `config.json`; `tensorrt-edgellm-export` (and the ONNX-less
`tensorrt-edgellm-build`) copy the deployed tables into the engine `config.json`; the
runtime uploads them to the attention plugin. Every key and flag carries its phase --
`prefill` or `decode` -- and nothing without a phase is read. Compared with a dense
deployment there is exactly one extra command:

```text
dense          HF checkpoint ──▶ export ──▶ build ──▶ run

skip-softmax   HF checkpoint ──▶ calibrate ──▶ export ──▶ build ──▶ run
                                 one-time,     copies the           reads the tables
                                 writes the    tables into          from the engine
                                 tables to     the engine           config
                                 config.json   config
```

### Workflow

**Step 1 -- calibrate the checkpoint.**

```bash
python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py calibrate \
  --model-dir /path/to/Qwen3-8B \
  --phase both \
  --max-seqlen 40960 \
  --bin-step 2048 \
  --cache-dir /path/with/room/modelopt-cache
```

`--phase prefill|decode|both` selects what is calibrated (default `prefill`).
`--max-seqlen` must cover the deployment context (calibration does not extrapolate).
`--bin-step N` calibrates a per-length table with one row every `N` tokens from
`--bin-min` (default 4096) up to `--max-seqlen`. Without `--target-sparsity`, the default
tier ladder (50 / 60 / 75 / 80%) is calibrated in one pass and the 75% tier is marked as
the deployed table. The checkpoint `config.json` then carries, for each calibrated
`<phase>` (`prefill`, `decode`):

| key | content |
|-----|---------|
| `skip_softmax_<phase>_scale_table` | the deployed table of that phase, rows of `[seqlen_bin, S]` |
| `skip_softmax_<phase>_scale_tables` | one table per tier, keyed `"50%"`, `"75%"`, ... |
| `skip_softmax_<phase>_target_sparsity` | the deployed tier |
| `skip_softmax_prefill_shape_advice` | the shape advisor's verdict (`on` / `off`) with the sparsity and mass share it was read from ([section 2.2](#skip-softmax-advisor)) -- informational, not read by export or build |

Abbreviated Qwen3-8B example (2k-step rows between the shown ones omitted):

```json
{
  "skip_softmax_prefill_target_sparsity": 0.78,
  "skip_softmax_prefill_scale_table": [
    [6144, 5757], [8192, 6380], [16384, 9398], [32768, 18012], [40960, 29741]
  ],
  "skip_softmax_decode_target_sparsity": 0.75,
  "skip_softmax_decode_scale_table": [
    [4096, 71], [32768, 212]
  ]
}
```

After a ladder calibration the tool automatically runs `price` (suppress with
`--no-price`), see [price](#skip-softmax-price). If the calibration cannot reach a tier,
the model has little to skip -- see [which checkpoints benefit](#which-models-benefit).

**Step 2 -- export and build.** No skip-softmax flag is needed once the checkpoint is
calibrated -- the exporter reads the tables from the checkpoint `config.json` and writes
them into the engine `config.json`:

```bash
tensorrt-edgellm-export /path/to/Qwen3-8B /tmp/qwen3_8b_onnx

./build/bin/llm_build \
  --onnxDir /tmp/qwen3_8b_onnx/llm \
  --engineDir engines/qwen3-8b-skip \
  --maxInputLen 40960 --maxKVCacheCapacity 40960 --maxBatchSize 1
```

The exporter deploys the checkpoint's tier (`skip_softmax_<phase>_target_sparsity`, else
its `skip_softmax_<phase>_scale_table`) for each phase on its own; a phase with no key
is dense. To pick another tier or force a phase dense without recalibrating, see
[overriding the checkpoint at export](#skip-softmax-export-flags).

**Step 3 -- run inference.** No runtime flag is needed: the engine `config.json` carries
the tables, the runtime reads them at load time and binds them to the attention plugin,
and the kernels decide per sequence from there:

```bash
./build/bin/llm_inference \
  --engineDir engines/qwen3-8b-skip \
  --inputFile input.json \
  --outputFile output.json
```

**Step 4 -- validate accuracy.** Compare the deployed engine against a dense build of the same checkpoint:

```bash
# dense reference: export with --skip-softmax-prefill-scale-table none
#                  --skip-softmax-decode-scale-table none, build, then
python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py evaluate \
  --model-dir /path/to/Qwen3-8B --engine-dir engines/qwen3-8b-dense \
  --llm-inference build/bin/llm_inference \
  --max-context 16384 --mmlu-samples 200 --save-results dense.json

python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py evaluate \
  --model-dir /path/to/Qwen3-8B --engine-dir engines/qwen3-8b-skip \
  --llm-inference build/bin/llm_inference \
  --max-context 16384 --mmlu-samples 200 --baseline dense.json --label "75%"
```

`evaluate` samples real RULER items filtered to the engine's context, optionally adds an
MMLU subset, and gates both deltas (default `--max-drop 0.03`); the exit code follows the
verdict so it can gate CI. `--arm prefill|decode|dense` scores one phase of an engine
that carries both tables, so each phase's accuracy cost is attributed on its own.

(skip-softmax-export-flags)=
**Optional -- override the checkpoint at export.** Two flags per phase; each phase is
resolved on its own, first match wins (`<phase>` = `prefill` or `decode`):

| export flag | checkpoint `config.json` key | effect |
|-------------|------------------------------|--------|
| `--skip-softmax-<phase>-scale-table table.json` | -- | that phase deploys the table as is; `none` (or `0`) makes that phase **dense** whatever the checkpoint says |
| `--<phase>-target-sparsity R` | `skip_softmax_<phase>_scale_tables` | the tier `R`'s table from the ladder |
| *(no flag)* | `skip_softmax_<phase>_target_sparsity`, then `skip_softmax_<phase>_scale_table` | the checkpoint's deployed tier, else its deployed table, else dense |

A phase with neither a flag nor a checkpoint key is dense. `--prefill-target-sparsity 0.5`
switches prefill to the 50% tier's table without recalibrating and leaves decode as the
checkpoint has it; `--skip-softmax-decode-scale-table none` keeps the prefill deployment
and runs decode dense. The ONNX-less builder (`tensorrt-edgellm-build`) takes the same
four flags and reads the same checkpoint keys with the same precedence.

(skip-softmax-price)=
**Optional -- price the tier ladder.** Not a step of the deployment: `price` measures every calibrated tier
against the dense build on the device that will run the engines: real-text TTFT and TPOT,
the attention kernel's own speedup and share of GPU time (nsys), and RULER / LongBench
drops per arm; `report` renders one or more runs into a single HTML page. Devices without
a Python export stack (Jetson, DGX Spark) get the exports from an x86 host first:

```bash
# x86 host with torch: write the dense + per-tier ONNX exports
python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py price \
  --model-dir /path/to/Qwen3-8B --work-dir /shared/qwen3-8b-price --export-only

# target device: build engines, measure, write price.json + skip_softmax_report.html
python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py price \
  --model-dir /path/to/Qwen3-8B --work-dir /shared/qwen3-8b-price \
  --lengths 1024 2048 4096 8192 16384 32768 40960 \
  --accuracy-lengths 8192 16384 32768 \
  --llm-build build/bin/llm_build \
  --llm-inference build/bin/llm_inference

# several models / platforms on one page
python kernelSrcs/fmha_cutedsl_blackwell/calibrate_skip_softmax.py report \
  --price-json /shared/qwen3-8b-price/price.json \
               /shared/llama-3.1-8b-price/price.json \
  --out skip_softmax_report.html
```

The result is recorded as `skip_softmax_price_list` in the checkpoint `config.json`.

**Re-targeting a deployed engine.** The tables are read from the engine `config.json` at
load time. Editing `skip_softmax_prefill_scale_table` or `skip_softmax_decode_scale_table`
there (or removing a key to run that phase dense) takes effect on the next launch with no
re-export or rebuild.

(skip-softmax-scope)=
### Scope, warnings and notes

**Scope.**

- **Phase**: prefill (context and chunked prefill) and vanilla decode, each with its own
  calibrated table. Speculative-decoding verify and draft steps run their normal kernels.
- **Kernels and architectures**: the Blackwell (SM100 / Thor) fused prefill kernels and
  the paged GQA decode kernel (SM100 / SM110) are the reference implementation, with the
  FMHA-v2 prefill kernel family following so the same `config.json` tables serve other
  architectures. Decode layers served by another decode kernel run dense.
- **Attention variants**: plain causal attention on FP16 scores. Sliding-window layers,
  FP8 KV-cache input and bidirectional / ViT attention run their normal kernels.
- **Quantization**: composable with FP8 / NVFP4 / INT4-AWQ GEMMs -- the skip decision is
  made on the attention scores, so the same attention kernel serves every GEMM precision.

**Warnings.**

- **Calibrate and gate per checkpoint.** The scale factor is a property of the model's
  score distribution -- checkpoints sharing an architecture can differ by an order of
  magnitude, and fine-tuned or reasoning-distilled variants tolerate far less sparsity
  than their base model. Never transfer a table between checkpoints.
- **Never deploy a flat $S$ to requests shorter than $S$.** With $\lambda = S / L$ a
  request with $L < S$ runs at $\lambda > 1$ ([section 2.1](#skip-softmax-table)). Use
  `--bin-step` calibration for mixed-length traffic, or keep the first bin above the
  shortest expected request.
- **Gate quantized engines with quantized engines.** Accuracy certified on an FP16 build
  does not transfer to an NVFP4 / FP8 build of the same checkpoint; run `evaluate` on the
  engine you deploy.
- **Thinking models need a matching evaluation protocol.** Bare-completion benchmarks score
  a reasoning model's thinking tokens, not its answer; evaluate with generation budgets
  that reach the answer.

**Notes.**

- **Real text for benchmarks**: random token ids give near-uniform scores and almost no
  skipped tiles. Use real prompts when measuring TTFT, and pin clocks on Thor
  (`jetson_clocks`) before any A/B.
- **Kernel-level detail**: the skip cubins, calibration internals and the CLI reference
  are documented in `kernelSrcs/fmha_cutedsl_blackwell/README.md`.
- **Calibration dependencies**: `torch`, `transformers`, `nvidia-modelopt`, `scipy`,
  `nltk`, `wonderwords`; point `--cache-dir` at a location with room for ModelOpt's RULER
  generation cache.
