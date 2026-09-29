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
| `--spec-draft-step` | Draft tokens proposed per round with `--mtp`. | `3` |
| `--context-reuse` | Run the engine with the managed context cache enabled. See *Context reuse*. | *(off)* |
| `--no-quantize-activations` | Export **both** sides with the activation Q-DQ dropped, isolating activation quantization as an error source. Diagnostic only: the engine is much slower. | *(off)* |
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
2. **Export** — `tensorrt_edgellm.scripts.export ... --num-decoder-layer N` exports only
   the first `N` layers to ONNX.
3. **Build** — `llm_build` builds the engine. The KV-cache capacity is kept small on
   purpose (see *Dump size*, below).
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

Only single-checkpoint MTP (Qwen3.5 and friends) is supported. The other variants name
specific target layers — `eagle3_target_layer_ids`, `dflash_target_layer_ids` — which a
truncation invalidates, so `--num-decoder-layer` still rejects them.

Expect acceptance to collapse to one token per round: the draft head was trained against the
full stack and a truncated base hands it a hidden state from the wrong depth, so its proposals
are almost always rejected. That costs speed, not correctness, and it does mean this test
exercises the commit path rather than deep multi-token rewinds.

---

## Scope

This is a base-model smoke test, under vanilla or MTP decoding, with or without context reuse.
It does not cover the other speculative-decoding variants (EAGLE / DFlash / DSpark), nor the
draft model's own numerics — the vanilla golden has no counterpart for those.
