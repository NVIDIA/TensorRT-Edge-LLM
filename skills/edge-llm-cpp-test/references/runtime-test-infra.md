<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Driving LLMInferenceRuntime in a test

`EngineExecutor` (`cpp/runtime/exec/engineExecutor.h`) is abstract so the runtime can be exercised without a
serialized engine on disk. A test injects a substitute through `ModelArtifacts::baseExecutor`; everything
else — deployment config, KV cache managers, tensor map, decoder registry — stays on the production path.

Scaffolding lives in `unittests/support/`:

| Header | Provides |
|--------|----------|
| `substituteEngine.h` | `MockEngineExecutor`, the tiny deployment configs, `writeModelDir()`, `makeVanillaArtifacts()`, `makeSpecArtifacts()`, `makeGreedyRequest()`, `makeGreedyBatch()` |
| `fakeEngineExecutor.h` | `FakeEngineExecutor` and `makeFakeEngine()` |
| `fakeState.h` | what a cache entry is (`FakeStateSlot`) and the arithmetic the engine runs on |
| `fakeCachePools.h` | `FakeKVGeometry`, `FakeKVWriter`, `FakeRecurrentWriter` — addressing and reading the pools |
| `fakeForwardLayout.h` | `FakeLayerComposition`, `FakeForwardLayout` — the topology and one forward's row layout |

## Choosing a substitute

**Default to `FakeEngineExecutor`.** It runs: it answers binding queries from a declared layer topology,
derives a value per token from the embedding the runtime looked up, carries it through every layer, stores
the result where the runtime says it goes, and decodes the logits that implies. Tests built on it assert
resulting state, so a refactor that changes when or how often the runtime calls the engine does not turn
them red.

**Reach for `MockEngineExecutor` only to verify interaction** — that a call happened, how many times, or in
what order relative to another. Its gmock actions can also stub values, but using it that way buys a
call-count assertion the test did not intend to make, which a scheduling change will break.

Both are permanent. Neither is a migration target for the other.

## Observing what the runtime decided

Much of this runtime's behavior is invisible in the response. A reused prefix and a recomputed one decode
the same text; a cache hit that saved no work looks identical to one that did.

The evidence is in the arguments to `prepare(profileIndex, dims, map, stream)` — which optimization profile,
how long a sequence, which pages, which state rows. `execute()` takes no arguments, so `prepare()` is the
only point where the runtime's own decisions are visible to a substitute.

`FakeEngineExecutor` records them:

| Call | Gives |
|------|-------|
| `preparedRounds()` | `{profileIndex, dims}` per `prepare()`, always collected |
| `recordBinding(name)` / `preparedBinding(name)` | INT32 binding contents per round; opt-in, one device-to-host copy per forward |
| `recordStatePool(name)` / `preparedStatePool(name)` | one value per row of a float state pool per round; opt-in |
| `forwardsRun()` | how many forwards this engine ran |
| `boundTensor(name)` | the tensor bound on the most recent `prepare()` |

Give each engine of a two-engine deployment its own instance and ask the one you mean. A shared list tagged
with an `isBase` flag only has to be filtered back apart.

Prefer `forwardsRun()` over counting `preparedRounds()` when the question is "did this engine do work":
nothing obliges the runtime to prepare exactly once per execute.

## What the engine produces, and how to override it

Left alone, the engine computes. A token's output depends on its own input and on every cached entry behind
it, read back through the page table the runtime just published — so a page reused when it should not have
been, dropped when it should have been kept, or mapped to the wrong sequence changes the tokens that come
back. Computed tokens are always at or above `kFirstComputedToken`, so they never collide with EOS.

The arithmetic is integers under a prime modulus below 256. That is not decoration: it is exact in fp16 and
bf16 and has no reassociation, so the same sequence gives bit-identical results however the runtime chose to
batch or chunk the work. Without that, an equivalence test would be comparing nearly-equal floats.

Three overrides, each replacing one thing and leaving the rest computed:

| Call | Replaces |
|------|----------|
| `forceOutputTokens(rows)` | the token each physical logits row decodes to, on every forward |
| `forceOutputTokensPerRound(rounds)` | the same, per `execute()` call; last entry repeats |
| `forceStateValues(generator)` | what gets stored in each cache entry, keyed by which entry it is |

A row past the end, or a negative id, is not overridden and keeps its computed token. Forcing replaces only
the last step: the forced token still carries into the next round's state, so a forced run stays internally
consistent.

The inner index is the **physical** row for that round, not a stable per-request identity: compaction moves a
surviving request to a lower row once another finishes.

Force output tokens when the test needs a specific decode schedule — EOS on round 3, a prompt that must
round-trip as literal text for a continuation. Force state values when the subject is *where* entries land
and the test wants entries it can name without running the arithmetic. Otherwise let it compute.

## A substitute must keep the contract of the engine it replaces

What a substitute stands in for is a compiled TRT engine. The arithmetic inside it is deliberately not
simulated — but the *contract of its bindings* is not arithmetic, it is interface, and the runtime is written
against it. Where a test uses an output at all, that output's meaning has to be the real engine's meaning.

The case that exists today: **a draft engine's `logits` binding holds log-probabilities; a base engine's holds
raw logits.** Every draft model ends in `F.log_softmax` (`tensorrt_edgellm/models/*/modeling_*.py`), so the
transform is inside the exported graph. The binding name is the same on both sides and nothing in `cpp/` says
which scale it carries.

Put a substitute on the wrong scale and speculative decoding breaks: drafting accumulates a step's values down
the chain, only a sum of log-probabilities is log P(path), and their non-positivity is what keeps a node's
score at or below its parent's — the topological order `constructVerificationDraftTree` assumes when it
searches only backwards for a node's ancestors. Raw logits are ordinarily positive, so the order inverts and
the verify tree collapses to one bit per row.

`FakeEngineExecutor::actAsDraftEngine()` and `makeLogProbDraftEngine()` select the draft scale. Use them for
any engine standing in for an EAGLE or MTP draft.

**The failure mode is worth knowing in its own right, because it does not look like a test bug.** The symptom
lands entirely inside runtime code that is correct, so it reads as a product defect, and it survives being
measured: the bindings really are wrong, the tree really does collapse, and every number you print confirms a
bug that is not there. Getting this wrong on this branch cost an issue filed against the runtime and a fix
that would have applied `log_softmax` twice.

**So when you add a substitute, or teach an existing one a new output, read the model's `return` rather than
the binding name.** The binding name tells you where a value goes, not what it means.

## Asserting on two runs instead of on a number

Do not assert that computed output equals a recomputation of it — mutate the arithmetic and both sides move
together. What the computed mode buys is **equivalence**: two runs that must agree, where the engine's only
job is to be sensitive enough that a difference shows.

- A prompt served partly from a reused prefix decodes what it decodes cold.
- A slot decodes the same tokens alone as it does beside another request. Assert **every** slot: two sequences
  sharing a row do not corrupt each other symmetrically, and one of them can look perfectly correct.
- Give the runs different inputs from each other. Two identical prompts agree under almost any bug.

The fixture has to be able to tell the runs apart before any of this means anything — state that as its own
assertion, so a fixture that has gone degenerate fails loudly instead of passing.

## Reading back cache contents

`FakeKVWriter::read()` and `FakeRecurrentWriter::read()` resolve an address through the page table the same
way the writers did, so a test can ask whether an entry survived without knowing where it was put. Reading a
specific value back is only meaningful alongside `forceStateValues()`; otherwise assert on the output.

Two traps:

- Layer indices are numbered **within a kind**. `past_key_values_2` is the third attention layer, not decoder
  layer 2.
- Recurrent state is observable only while its sequence is live. The runtime releases the resident row on
  completion, so a pool read after `handleRequest()` returns finds it cleared. Use `recordStatePool()` to
  capture it at `prepare()` time, which asserts the better property anyway: that the state carried across a
  forward boundary.

## Scope

`FakeEngineExecutor` covers the core decode path of attention, recurrent and hybrid decoders. Bindings
specific to speculative decoding, vision, audio and LoRA are reported absent, as `MockEngineExecutor` reports
them too. A test needing one of those either extends the fake or is a genuine case for the mock.
