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
| `--atol` | `allclose` absolute tolerance (reported per round; does not gate). | `2e-2` |
| `--rtol` | `allclose` relative tolerance (reported per round; does not gate). | `2e-2` |
| `--max-input-len` | Engine `maxInputLen`. Engine build parameters can affect accuracy. | `128` |
| `--max-kv-cache-capacity` | Engine `maxKVCacheCapacity` (drives the dump size; see *Dump size*). | `256` |
| `--keep` | Keep the temporary work directory (golden / engine / dump) for inspection. | *(off)* |

The script's exit code is the PASS/FAIL gate (`0` = pass). Use it both locally during
development and as the body of the CI test.

It also accepts `--build-dir` and `--python` to point at a non-default build dir or
interpreter (handy when working across machines); they default to `<repo>/build` and
`python3` and are omitted here to keep the common case simple.

Each round prints the minimum cosine (and the tensor that hit it), the maximum absolute
difference, and an `allclose` flag at the given `--atol` / `--rtol`; the final line reports
the worst cosine over the whole run alongside the thresholds. `--verbose` adds a per-tensor
breakdown.

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
`maxInputLen` / `maxKVCacheCapacity` small (the validation uses short, fixed prompts),
which keeps the dump at a few hundred MB rather than multiple GB. If you point the script
at long prompts, raise both — but note the dump grows with the cap.

---

## Scope

This is a base-model + vanilla-decoding smoke test. It does not cover the
speculative-decoding variants (EAGLE / MTP / DFlash); combining `--num-decoder-layer` with
those is rejected.
