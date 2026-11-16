# Few-Layer Numeric Validation

This guide explains the **few-layer numeric validation** — an early-fail smoke test
that checks whether the EdgeLLM engine still produces the right numbers, without
waiting for a full model run.

It runs only the **first N decoder layers** (default 4) of a checkpoint through two
producers and compares them:

1. a **PyTorch golden** (a high-confidence reference, not an oracle), and
2. **EdgeLLM engine inference**,

over a prefill plus a few greedy decode rounds. It compares per round, per sequence,
per layer:

- **logits** (last real token),
- **KV cache** for attention layers, and
- **recurrent + conv state** for Mamba / Gated-DeltaNet hybrid layers.

A numeric regression then shows up in seconds instead of after a full generation.

---

## Quick Start

Assuming you are in the repo working directory with EdgeLLM built (`build/`) and its
venv active:

```bash
./scripts/few-layer-validation.sh \
  --model /path/to/checkpoint \
  --num-layers 4 \
  --cos 0.99
```

| Flag | Meaning | Default |
| --- | --- | --- |
| `--model` | HuggingFace / ModelOpt checkpoint directory (or HF repo id). | *(required)* |
| `--num-layers` | Number of leading decoder layers to validate. | `4` |
| `--cos` | Minimum per-tensor cosine similarity; the run fails below it. | `0.99` |
| `--scale-tol` | Maximum per-tensor `\|norm(edgellm)/norm(golden) - 1\|`; the run fails above it. See *Why two gates*. | `0.25` |
| `--atol` | `allclose` absolute tolerance (reported per round; does not gate). | `2e-2` |
| `--rtol` | `allclose` relative tolerance (reported per round; does not gate). | `2e-2` |
| `--max-input-len` | Engine `maxInputLen`. Engine build parameters can affect accuracy, so it stays overridable. | *(from the prompts)* |
| `--max-kv-cache-capacity` | Engine `maxKVCacheCapacity` (drives the dump size; see *Dump size*). | *(from the prompts)* |
| `--mtp` | Also build the checkpoint's MTP draft and run the engine with speculative decoding. See *Speculative decoding*. | *(off)* |
| `--dflash-draft-dir` | DFlash draft checkpoint (a separate, untruncated checkpoint). Requires `--direct-builder`. See *Speculative decoding*. | *(off)* |
| `--eagle3-draft-dir` | EAGLE3 draft checkpoint, same shape as `--dflash-draft-dir`. | *(off)* |
| `--spec-draft-step` | Draft tokens proposed per round with `--mtp`. | `3` |
| `--context-reuse` | Run the engine with the managed context cache enabled. See *Context reuse*. | *(off)* |
| `--no-quantize-activations` | Export **both** sides with the activation Q-DQ dropped, isolating activation quantization as an error source. NVFP4 linears then run through the NVFP4-A16 GEMM plugins. | *(off)* |
| `--trt-native-moe` | Build the MoE experts through TensorRT's native MoE layer instead of the Edge-LLM MoE plugins, on whichever frontend builds the engine (ONNX export or `--direct-builder`). The golden is arm-agnostic, so the same reference gates both arms. | *(off)* |
| `--trt-native-moe-w4a4` | With `--trt-native-moe`: export NVFP4 experts as TensorRT MoE quantization mode 2 (W4A4: NVFP4 dynamic activation quantization, FP4 weights with separate block and global scales) instead of W4A8. Export-only, so it is refused together with `--direct-builder`. Needs TensorRT 11.6 or newer (`IMoELayer::setQuantizationDynamicDblQ`); the few-layer CI cases (Qwen3-30B-A3B, Nemotron-3-Nano) carry `minimum_trt_version: "11.6"` and are skipped on older runners. | *(off)* |
| `--direct-builder` | Build the engine with the checkpoint-direct builder (`edgellm build`) instead of ONNX export + `llm_build`: stages 2 and 3 become one step, compiled for the **active** GPU (`--target-sm` is not used), and stage 4 passes `--checkpointDir` so `llm_inference` reads the weights the builder leaves in the checkpoint. Required for every architecture the ONNX path cannot build (see *Architecture coverage*). Not combined with `--mtp`. | *(off)* |
| `--no-golden-cache` | Ignore the shared golden cache under `$EDGE_LLM_CACHE_DIR/golden_layers` (see *PyTorch golden*) and always regenerate locally without uploading. | *(off)* |
| `--keep` | Keep the temporary work directory (golden / engine / dump) for inspection. | *(off)* |

The script's exit code is the PASS/FAIL gate (`0` = pass). Use it both locally during
development and as the body of the CI test.

It also accepts `--build-dir` and `--python` to point at a non-default build dir or
interpreter (handy when working across machines); they default to `<repo>/build` and
`python3` and are omitted here to keep the common case simple.

Each round prints the minimum cosine and the worst norm ratio (each with the tensor that hit
it), the maximum absolute difference, and an `allclose` flag at the given `--atol` / `--rtol`;
the final line reports the worst of both over the whole run alongside the thresholds.
`--verbose` adds a per-tensor breakdown.

### TRT-native MoE arms (TensorRT >= 11.3, SM 10.x / 11.x)

`--trt-native-moe` needs a TensorRT >= 11.3 wheel and libraries on the path and a
Blackwell-class GPU (SM 10.x / 11.x). It works on either frontend:

```bash
# ONNX frontend: edgellm export --trt-native-moe + llm_build
./scripts/few-layer-validation.sh \
  --model /path/to/nvidia-Qwen3-30B-A3B-NVFP4 \
  --num-layers 2 --cos 0.96 --target-sm 100 --trt-native-moe

# Checkpoint-direct frontend: edgellm build --trt-native-moe
# (compiles for the active GPU; --target-sm is not used)
./scripts/few-layer-validation.sh \
  --model /path/to/nvidia-Qwen3-30B-A3B-NVFP4 \
  --num-layers 2 --cos 0.96 --direct-builder --trt-native-moe
```

The golden does not depend on the MoE arm, so the plugin build of the same checkpoint
and both TRT-native builds are gated against one reference.

---

## Why two gates

Cosine measures direction and is **scale-invariant**: if one side drops a scale factor —
both implementations computing the same thing up to a constant — cosine still reads 1.0.
That has happened before, so the run also gates on the norm ratio
`||edgellm|| / ||golden||`, which is exactly the degree of freedom cosine throws away.

The ratio is a usable gate because it is far tighter than cosine on a quantized model: both
sides carry same-strength quantization noise, which largely cancels in a ratio, whereas a
dropped factor is common-mode and does not cancel. Measured on Qwen3-0.6B over 4 layers:

| | worst cosine | worst `\|ratio - 1\|` |
| --- | --- | --- |
| FP16 | 0.99998 | 9e-4 |
| NVFP4, KV / state | 0.968 | 1e-2 |
| NVFP4, logits | 0.975 | 1e-1 |

Logits are the loose case: after the final norm their magnitude depends only on the hidden
state's *direction*, so it moves much more than a cache tensor's does. The `0.25` default
clears that with room to spare while still catching any realistic factor bug — a dropped
`2x`, `0.5x` or `sqrt(head_dim)` is a 100%+ deviation. If you gate only on cache/state
tensors, `--scale-tol 0.05` is achievable and catches finer drift.

`allclose` is reported but deliberately does not gate: absolute error is tied to each
tensor's dynamic range, and that spans orders of magnitude across layers (a Qwen3 K cache
carries outlier channels ~100x the bulk), so no single tolerance fits every tensor.

### Where each case's cosine threshold comes from

An fp16 or GDN-hybrid model holds ~0.9999 and is gated at `0.99`.

**Quantized (NVFP4) drops to ~0.97**, for a dense model as much as a recurrent one. The
E2M1 grid is coarse enough that a one-fp16-ULP difference flips an element to the
neighbouring level, amplifying it ~15x per quantized linear, and three of those sit in
series per decoder layer. Two numerically equivalent implementations therefore decorrelate
within a few layers. Only a finer quantization step tightens this; a more faithful golden
does not, which is what `--no-quantize-activations` demonstrates (it reaches ~0.99998).
A model whose leading layers are recurrent puts fewer quantized linears in series and
lands higher, so its threshold is set by headroom across arches rather than by need.

**MoE drops further, and for a different reason: routing is a discrete decision.** The
router's top8-vs-top9 margin has a heavy left tail (median 0.06 sigma of the router
logits, 5th percentile 0.004), so at the agreement the two sides already have on the
hidden state (0.99999, i.e. relative L2 4e-3) roughly 1% of token-layer pairs sit inside
that margin and each side legitimately picks a different expert set. One such swap costs
that token's logits ~0.999 cosine; a few stacking along a token's path reach ~0.995.
Which tokens flip is not reproducible, since neither side is bit-reproducible across runs
(TensorRT re-picks tactics per build, and the reference MoE sums experts with an atomic
`index_add`). It lands almost entirely in the logits: KV and recurrent state stay above
0.99, because one token's flip is diluted across the whole cache tensor. Recipe-specific
MoE gates therefore range from `0.95` to `0.98`; wide-vocabulary GPTQ logits use `0.96`.

---

## What it does

The script runs five stages, all into a temporary work directory:

1. **PyTorch golden** — runs the first `N` layers of the checkpoint and dumps per-round
   logits + KV / recurrent / conv state, plus the token sequence it samples
   (`tests/golden_layer_dump.py`). For quantized checkpoints the recipe-quantized
   projections are swapped for fake-quant linears that reuse the same ops the export
   emits, so the golden matches the engine's quantized numerics
   (`tests/golden_quant_linears.py`).

   When `$EDGE_LLM_CACHE_DIR` is set, this stage first checks a shared cache at
   `$EDGE_LLM_CACHE_DIR/golden_layers/`, keyed by model + the options that change the
   golden's content (layer count, requests file, activation quantization) — **not** by
   PyTorch/Transformers version, since teacher-forcing (see *Runtime hooks* below) makes
   library-version reproducibility unnecessary. A hit skips regeneration entirely; a miss
   regenerates locally and uploads the result for the next run/platform. This is what lets
   PyTorch-less runners (e.g. Drive Thor) validate without installing PyTorch + Transformers,
   as long as some earlier run has already populated the cache for that model. Pass
   `--no-golden-cache` to always regenerate locally.
2. **Export** — `tensorrt_edgellm.onnx.cli ... --num-decoder-layer N` exports only
   the first `N` layers to ONNX.
3. **Build** — `llm_build` builds the engine. The KV-cache capacity is kept small on
   purpose (see *Dump size*, below). With `--direct-builder`, stages 2 and 3 are one
   step: `edgellm build ... --num-decoder-layer N` builds the engine
   straight from the checkpoint for the active GPU (no ONNX), and stage 4 passes
   `--checkpointDir` so `llm_inference` can read the weights the builder leaves in
   the checkpoint. Stages 1, 4 and 5 are unchanged.
4. **Inference + dump** — `llm_inference` runs with the dump enabled and is teacher-forced
   with the golden's token sequence (see *Runtime hooks*), so both sides decode the
   identical sequence; it writes a single safetensors file of all rounds.
5. **Compare** — `tests/compare_layer_dumps.py` reconciles the two dumps and checks every
   tensor against the cosine threshold.

---

## Runtime hooks

The dump and teacher-forcing are driven by environment variables and are **no-ops unless
set**, so they add no overhead to a normal run:

| Variable | Effect |
| --- | --- |
| `EDGELLM_DUMP_LOGITS_KVCACHE_LAYERS` | Number of leading decoder layers `k` to dump (dumps layers `0..k-1`). |
| `EDGELLM_DUMP_LOGITS_KVCACHE_DIR` | Output directory for the dump file. |
| `EDGELLM_FORCE_TOKENS_FILE` | Optional teacher-forcing: a file of per-sequence token ids. When set, the decode loop replays these tokens instead of its own sampled ones, so the run follows the golden token-for-token. |
| `EDGELLM_IGNORE_EOS` | Force a fixed generation length (do not stop at EOS). |

The two `EDGELLM_DUMP_LOGITS_KVCACHE_*` variables are XOR-coupled: setting exactly one is
an error. Teacher-forcing is only active alongside a dump and logs a warning, since it
overrides the model's own sampled tokens.

### Why teacher-forcing?

Without it, each side greedily samples its own next token. Greedy decoding is sensitive to
near-tie argmax flips: a tiny numeric difference can pick a different token, after which the
two sides decode unrelated sequences and every later round compares mismatched states — the
comparison stops meaning anything.

Teacher-forcing feeds the golden's tokens to the engine so both sides process the *same*
sequence. The handoff is automatic: the golden (stage 1) records the tokens it samples, the
script writes them to a force-tokens file, and the engine (stage 4) replays them via
`EDGELLM_FORCE_TOKENS_FILE` — you never write that file by hand. The dump still records the
token the engine *would* have sampled, so a genuine divergence is still visible.

---

## Dump size

The engine dumps each per-layer tensor **full-length** over the active batch (a plain
copy) and lets the comparison slice each sequence to its valid length in PyTorch — this
keeps the C++ side simple. The KV-cache sequence dimension therefore equals
`maxKVCacheCapacity`, so the dump size scales with it directly. The script keeps
`maxInputLen` / `maxKVCacheCapacity` no larger than the prompts need — each is derived
from the sequence the golden tokenized, rounded up to a KV page and floored at the old
`128` / `256` — which keeps the dump at a few hundred MB rather than multiple GB. Both stay
overridable, but note the dump grows with the cap.

Deriving them rather than hard-coding them is deliberate: the requests JSONs are shared
fixtures, and a case that pins its engine bounds breaks the day someone lengthens one.

---

## Wall clock

Every stage prints its own wall clock and the script ends with a summary table, so a CI
timeout points straight at the stage that grew rather than at the run as a whole.

Export and build dominate everywhere, and both scale with the **checkpoint**, not with
`--num-layers`: a 4-layer slice of a 256-expert MoE still loads and repacks every expert
weight in those layers. That is why the CI test carries a per-model `timeout`
(`tests/defs/test_few_layer_validation.py`) instead of one global value. A cold first read
of a large checkpoint can easily double the export stage, so size a new case's timeout
from a cold run.

The export skips a checkpoint's vision / audio towers (`--skip-visual --skip-audio`):
only the LLM backbone is compared, so building the encoders would be pure cost.

---

## Request fixtures

`--input-file` takes an `llm_inference` requests JSON, and the script reads `batch_size` and
`max_generate_length` out of it so the golden and the runtime stay aligned. The choice is a
coverage decision, not a cosmetic one:

| Fixture | Shape | When |
|---|---|---|
| `tests/test_cases/ragged_batch.json` | 4 requests, prefills 17-57 tokens, 6 rounds | The default, and what a new case should use. |
| `tests/test_cases/llm_few_layer.json` | 1 request, 8 rounds | A checkpoint too large to push four sequences through the PyTorch golden inside the case's timeout. |
| `tests/test_cases/llm_context_reuse.json` | 3 shared-prefix requests, `batch_size: 1` | `--context-reuse` only — see below. |

Two reasons to prefer the ragged default.

**A `batch_size: 1` fixture compares one row.** The engine dumps each tensor full-length over
the active batch and the comparison slices per sequence, so a single-request fixture never
distinguishes "read slot 0" from "read the right slot", and never prefills two different
lengths in one call. Four unequal prefills cost no extra build and no extra golden checkpoint
load.

**A long `max_generate_length` buys nothing.** Both gates are a min over every tensor and every
round, so each additional decode round is another chance for a single routing-boundary token to
become the reported score, with no new code exercised. A 128-round fixture is how one MoE case
ended up gated at 0.90 on the strength of one round out of 128.

**Moving an existing case onto it will lower its measured floor.** Four sequences put far more
tokens through the model than one did, and on a quantized MoE every extra token is another draw
from the expert-routing boundary that the gate takes a minimum over. Three cases that read
0.97-0.99 on the single-request fixture read 0.93-0.97 on this one, with the worst value landing
on a batch slot the old fixture never had. That is the fixture finding a floor that was always
there, not a regression -- but it means the gate has to be re-measured, not carried across.

Read the reproducibility before setting the new number. A value that repeats to the digit across
two CI runs is a deterministic floor and needs only a little headroom; one that moves by a couple
of points between runs is routing-driven and needs a full spread below the lowest sample. The
same two cases can both sit 0.002 from their gate and mean completely different things.

The `engine:` block on a few-layer case does not set the engine bounds: the script derives
`maxInputLen` / `maxKVCacheCapacity` from the prompts (see **Dump size**).

---

## Context reuse

`--context-reuse` enables the managed context cache while the golden prefills every request in
full. That is the invariant: a prefix restored from the cache has to equal one that was just
computed, so a full-prefill golden is the reference.

Point `--input-file` at requests that share a long prefix and set `batch_size` to `1`, so each
one is issued as its own request — the first populates the cache and the rest hit it.
`tests/test_cases/llm_context_reuse.json` is such a file. The engine writes one dump per request
and the comparison consumes them in request order, mapping each onto the golden row that ran the
same prompt.

Two details the dump has to get right, both of which only bite once a prefix is actually reused:

- **The KV pool is gathered through the page table.** The pool doubles as a
  `[maxBatch, capPadded, heads, dim]` view, but a sequence's tokens are contiguous in it only
  while the page table is the identity. Reuse hands a sequence pages that belong to an earlier
  request, so the dump walks its logical pages instead of reading a slot-sized span.
- **The reused prefix is added back to `context_lengths`.** A reusing request only executes the
  suffix after the cached prefix, so the runtime's token list is shorter than the cache actually
  is. The prefill dump records the difference and every later round adds it back.

Expect the reusing requests to agree with a full prefill but not bit-exactly on a hybrid model:
a recurrent state resumed from a snapshot processes the suffix as its own chunk, so the scan's
summation order differs. Measured on Qwen3.5-0.8B over 4 layers, that is worth about 2e-5 of
cosine — far inside the `0.99` gate. Attention-only layers do come out bit-identical.

---

## Speculative decoding

`--mtp` runs the engine with MTP speculative decoding while the PyTorch golden stays
**vanilla** — that is the point, not an approximation. Speculative decoding is defined by
output equivalence: whatever the draft proposes, the base model's committed KV / recurrent
state and its last-token logits must equal what plain autoregressive decoding would produce
for the same tokens. So a vanilla golden is exactly the right reference, and the comparison
becomes a check on the accept / rewind / commit machinery.

Two things follow from a round committing several tokens at once:

- **Rounds no longer line up.** Each `(engine round, sequence)` is paired with the golden
  round holding the same committed length, read from the dumped `context_lengths`. Under
  vanilla decoding every row advances by one per round and this degenerates to `g == r`, so
  it is a single code path.
- **Teacher-forcing trims instead of overwriting.** The vanilla path overwrites the sampled
  token, which is safe because that token has no cache entry yet. In a speculative round only
  the *last* committed token is in that position — the earlier ones already have cache entries
  written from the draft's proposals. So on the first token that disagrees with the golden the
  acceptance is trimmed, which drops that token's cache entry, and only then is the token
  replaced. This happens before the KV-cache commit.

MTP is the straightforward case: the draft reads the base's **last** hidden state, so a
truncation moves which layer that is but never leaves it undefined.

EAGLE3 and DFlash need more care, because their drafts read **several** specific depths of
the full base stack — `eagle3_target_layer_ids`, `dflash_target_layer_ids`. Those ids are
trained against the full stack, so a truncation invalidates them outright. Passing
`--eagle3-draft-dir` or `--dflash-draft-dir` clamps each id into the truncated range instead
of rejecting the combination. The two differ in what the runtime demands of the result:
DFlash only range-checks, so duplicates after clamping are accepted, while EAGLE3's
`validateEagleConfig` requires the three conditioning layers to stay **distinct**, so a
plain `min()` clamp is not enough and collisions are nudged to the nearest free slot. Fewer
than three decoder layers cannot satisfy that at all and is rejected up front.

The draft checkpoint is separate from the base and is never truncated — for EAGLE3 it is
architecturally a single fused layer whatever the base's depth.

Clamped ids mean the draft is reading hidden states from depths it was not trained on. As
with a truncated base generally, that costs acceptance rate, not correctness: this test
covers the base's committed verify / accept / commit output, not the draft's own numerics.

DSpark and JetSpec are not covered.

Expect acceptance to collapse to one token per round: the draft head was trained against the
full stack and a truncated base hands it a hidden state from the wrong depth, so its proposals
are almost always rejected. That costs speed, not correctness, and it does mean this test
exercises the commit path rather than deep multi-token rewinds.

---

## Architecture coverage

EdgeLLM supports far more checkpoints than this test has been exercised against. The two
sets are not the same thing: a model is covered here only once the *truncation* machinery
works for it end to end — the golden can load `--num-decoder-layer` of it, the builder can
build the truncated graph, and the comparison passes. Absence from this table means nobody
has tried, not that the model is unsupported.

| `model_type` | Checkpoint (layers) | Flags beyond `--model` / `--num-layers` | Worst cosine | Gate |
| --- | --- | --- | --- | --- |
| dense decoder | Qwen3-0.6B (4) | — | 0.99998 | `0.99` |
| nemotron_h | NVIDIA-Nemotron-3-Nano-4B-NVFP4 (4) | — | — | `0.96` |
| gated-deltanet hybrid | Qwen3.5-0.8B (4) | — | — | `0.99` |
| gemma4_unified | gemma-4-12B-it (4) | `--direct-builder` | 0.99991 | `0.99` |
| nemotron_h MoE-hybrid | NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 (6) | `--direct-builder` | 0.96217 | `0.95` |
| nemotron_h MoE-hybrid, TRT-native MoE | NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 (6) | `--trt-native-moe` | 0.96163 | `0.95` |
| nemotron_h MoE-hybrid, TRT-native MoE | NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (6) | `--trt-native-moe` | 0.96815 | `0.95` |
| qwen3_moe | Qwen3-30B-A3B-GPTQ-Int4 (4) | `--direct-builder` | 0.99859 | `0.98` |
| qwen3_5_moe (NVFP4) | Qwen3.6-35B-A3B-NVFP4 (4) | `--direct-builder` | 0.926-0.948 | `0.90` |
| qwen3_5 dense | Qwen3.8-27B-NVFP4 (4) | `--direct-builder` | 0.99892 | `0.96` |
| qwen3_5 dense, NVFP4-A16 Marlin | Qwen3.6-27B-NVFP4 (4) | `--direct-builder --no-quantize-activations` | 0.99934 | `0.99` |
| qwen2_5_vl | Qwen2.5-VL-3B-Instruct (4) | `--direct-builder` | 0.99689 | `0.99` |
| internvl3_5 | InternVL3-1B-hf (4) | `--direct-builder` | 0.99998 | `0.99` |
| dflash | Qwen3-4B + Qwen3-4B-DFlash-b16 (4) | `--direct-builder --dflash-draft-dir` | 0.99995 | `0.99` |
| eagle3 | Qwen3-4B + Qwen3-4B_eagle3 (4) | `--direct-builder --eagle3-draft-dir` | 0.99995 | `0.99` |

Notes on individual rows:

- **gemma-4-12B-it is `gemma4_unified`, not the `gemma4` of gemma-4-26B-A4B**, so the
  two are not redundant. Its leading layers are sliding-window attention and run
  through the CuTe DSL FMHA-v2 vision-block kernel, which nothing else here reaches.
- **Six layers for the nemotron_h MoE-hybrid** is not arbitrary: it lands on a
  `M E M E M *` window, so one run covers Mamba, MoE and attention together. Four would
  stop short of the attention layer.
- **The nemotron_h `--trt-native-moe` rows** (ReLU² experts through TensorRT's non-gated
  `TRT_MoE`, W4A8 for the NVFP4 checkpoint, FP16 for the BF16 one) were measured on an RTX
  PRO Blackwell (SM 12.0) with a TensorRT 11.5 build; the worst
  tensor is the same Mamba `conv_state` as in the plugin rows, and the plugin arms read
  0.95455 (NVFP4, SM 12.0) and 0.96816 (BF16) on the same box.
- **No row here covers a vision tower.** The fixture sends text-only requests and
  compares decoder layers, so a vision-language row establishes that the model's *text
  backbone* truncates and computes correctly — nothing more. Several checkpoints in the
  table above (`gemma-4-12B-it`,
  `Qwen3.6-35B-A3B`, `Qwen3.8-27B`) carry a `vision_config` for the same reason and
  under the same limit. Covering a visual encoder needs a golden that accepts image
  input, which this test does not have.
- **qwen3_vl is not in the table because this script cannot build it.** Its LLM engine
  declares deepstack visual-feature injection points whether or not a request carries an
  image, so it refuses to load without a companion visual engine, and the driver builds
  the LLM component only. qwen2_5_vl and internvl3_5 have no such requirement.
- **internvl3_5 has only ever been validated through `--direct-builder`.**
  `tensorrt_edgellm/_export_api.py` registers no `internvl` / `internvl_chat` handler, so
  the ONNX path would fall back to the generic decoder; that path is untested here.
- **A row in this table is not a CI case.** CI runs few-layer validation on large
  checkpoints only -- roughly 10B and up. A model small enough to run the full
  export/build/inference suite is already covered there, so a few-layer case for it spends
  CI time without adding coverage. The smaller rows here (Qwen3-0.6B, Nemotron-3-Nano-4B,
  Qwen3.5-0.8B, the two VLMs, and both speculative-decoding rows) record that the
  truncation machinery works for that `model_type`, which is what a future large case for
  the same architecture starts from. They are run by hand.
- **The DFlash row pairs the b16 draft with `Qwen/Qwen3-4B`**, while the support matrix
  pairs it with `Qwen/Qwen3-4B-Instruct-2507`. The validated combination is the former.
- **The Qwen3.6 row is the one loose gate here, and the range in its column is not a
  typo.** Six repeats landed between 0.926 and 0.948, every one of them on the logits
  of the same decode round: that token sits on an expert-routing boundary, so the two
  sides legitimately choose different experts and the value moves run to run. The rest
  of the run is healthy — median 0.991, KV tensors 0.991. A single ambiguous token
  setting the gate for a whole run is a design limitation of gating on a minimum,
  tracked separately; do not lower this further to make a run pass.
- **`Qwen3.8-27B-NVFP4` is not a pure NVFP4 checkpoint** despite the name: the golden
  patches 13 NVFP4 and 13 FP8 linears. The FP8 half explains why a quantized dense
  model lands at 0.999 rather than the ~0.97 the section above predicts.
- **The `Qwen3.6-27B-NVFP4` NVFP4-A16 row is the only CI case for the dense
  NVFP4-A16 Marlin GEMM.** It reproduced 0.99934 to the digit across four builds, worst
  on the FP8 KV cache. Its gate does not separate A16 from W4A4 (0.99875 on the same
  checkpoint): it catches a broken Marlin epilogue — dropping the global scale turns
  every tensor into NaN — not a routing fallback. The first four layers carry no
  massive activation, so it also cannot see the fp16 accumulator overflow that
  Qwen3-0.6B-NVFP4 `layers.2.mlp.down_proj` triggers.
- **The two MoE rows build only through `--direct-builder`.** The ONNX path fails
  for both, in different places: a Mamba `in_proj` Gemm rank mismatch for nemotron_h, and
  `QkvConcatPlugin::getOutputShapes` on the INT4 GPTQ attention path for qwen3_moe.

The measured column is the worst value seen on a Blackwell-class runner, not a contract —
the gate is. A dash means the case predates this table and no figure was recorded with it.

Treat the margin as informative. The MoE rows in particular move with kernel changes: both
dropped once across an upstream rebase (by 0.005 and 0.002) and then reproduced to five
decimal places across the next one. A case sitting close to its gate deserves an
investigation into what moved, not a looser gate.

A case's `requirements.min_gpu_memory_gb` is not the reduced checkpoint's size for a
quantized model. The golden dequantizes NVFP4 weights eagerly, and the dequantizer
materializes one float per element, so a wide `lm_head` costs several GiB of temporaries
that the checkpoint-size rule does not see. `Qwen3.8-27B-NVFP4` computes to 9 GiB by that
rule and needs more than 32. Derive the number from a run rather than from the file sizes
whenever the checkpoint is quantized.

Only a subset of these run in CI; those cases live under `tests/model_configs/l0/`. The
rest are kept runnable by hand through the flags above, which is why this table exists.

When choosing what to add or drop from CI, weigh the dimension a case covers, not the size
of its checkpoint. The DFlash and EAGLE3 rows are each the only case exercising a decoding
mode; dropping either takes that dimension to zero while the remaining cases still look
plentiful.

Weigh it honestly in both directions. `Qwen2.5-VL-3B-Instruct` was registered and then
removed on exactly this test: it was added to cover a vision-language family, and since
no row here reaches a vision tower, it was buying coverage of a text backbone already
covered several times over. A case that only appears to cover a dimension is worse than
no case, because it stops anyone else from adding the one that would.

---

## Scope

This is a base-model smoke test, under vanilla, MTP, EAGLE3 or DFlash decoding, with or
without context reuse. It does not cover DSpark or JetSpec, nor any draft model's own
numerics — the vanilla golden has no counterpart for those.
