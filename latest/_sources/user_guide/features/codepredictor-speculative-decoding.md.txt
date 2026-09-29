# CodePredictor Speculative Decoding

## Overview

Audio generation in Qwen3-Omni and Qwen3-TTS has two nested autoregressive loops. The outer one
is the Talker, which emits the first residual-VQ (RVQ) code of each audio frame. The inner one is
the **CodePredictor**, which then emits the remaining `num_code_groups - 1` codes of that same
frame, one sequential forward pass each. Because the inner loop runs many times per frame while
the outer loop runs once, the CodePredictor dominates audio decode time.

This feature speculates on that inner loop. Unlike the draft-model modes documented in
[Speculative Decoding](../examples/speculative-decoding.md), it needs **no draft model, no
training, and no extra weights**: the draft comes from parameters the checkpoint already ships.

**Key points**:

- The draft is free — the CodePredictor already carries one `lm_head` per RVQ depth over a shared
  residual stream, so a head for a later depth applied to the current hidden state is a valid
  one-step-ahead proposal
- Verification is exact speculative sampling, so the sampled output distribution is unchanged
- Off by default; enabled per run with a single flag, and no separate engine build step
- Degrades to the autoregressive loop with a warning rather than mis-decoding when the model or
  engine cannot support it

## Why the draft is free

Standard speculative decoding needs a second, cheaper model that approximates the target. Here
the approximation already exists inside the target.

The CodePredictor decodes the RVQ depths of one frame from a single residual stream: at depth
`s` it holds a hidden state `h_s`, and it owns a separate `lm_head` for every depth. The
autoregressive loop only ever applies `head_s` to `h_s`. But because all the heads read the same
stream, applying `head_{s+t}` to `h_s` produces a usable distribution for the depth `t` steps
ahead — it is what the model would predict if the intervening codes carried no new information.

That makes a draft cost exactly one GEMV per drafted depth, against a full decoder forward pass
for the target. There is nothing to train, quantize, distil, or ship: the heads are already in
the checkpoint, and older engines gain the capability without being re-exported.

The proposal quality falls off as `t` grows, which is why acceptance saturates after a few
depths rather than improving indefinitely.

## Usage

Speculation is off by default. Enable it by passing a verify window:

```bash
# Qwen3-Omni
./build/examples/llm/llm_inference \
    --engineDir $ENG/thinker \
    --multimodalEngineDir $ENG/multimodal \
    --enableAudioOutput \
    --talkerEngineDir $ENG/talker \
    --code2wavEngineDir $ENG/multimodal/code2wav \
    --inputFile $WORKSPACE_DIR/input.json \
    --outputFile $WORKSPACE_DIR/output.json \
    --outputAudioDir $WORKSPACE_DIR/audio \
    --cpSpecVerifySize 3

# Standalone Qwen3-TTS
./build/examples/omni/qwen3_tts_inference \
    --talkerEngineDir $ENG/talker \
    --code2wavEngineDir $ENG/code2wav \
    --tokenizerDir $ENG/talker \
    --inputFile $WORKSPACE_DIR/input.json \
    --outputFile $WORKSPACE_DIR/output.json \
    --outputAudioDir $WORKSPACE_DIR/audio \
    --cpSpecVerifySize 3
```

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--cpSpecVerifySize N` | `0` (off) | Verify window: one committed RVQ depth plus `N-1` drafted depths. Valid range 2-8; `1` is rejected because it leaves no draft slot. |

Acceptance saturates once the window passes four depths, so the useful settings are the narrow
ones — a wider window spends verification time without returning more codes.

Engines need no special build step. The builder recognises a CodePredictor from the `model` field
in its engine config and widens the upper bound of that engine's generation profile so one verify
pass can bind several positions. The profile's optimization shape stays at the single-token decode
shape, so tactic selection is unchanged for the default autoregressive path. No other engine is
affected.

## How a round works

One round commits at least one RVQ code and at most `N`:

1. **Draft.** For each of the `N-1` positions ahead of the committed depth, apply the
   corresponding `lm_head` to the current hidden state and sample a proposal. All positions are
   scored in one batched GEMV.
2. **Verify.** Run one CodePredictor forward pass over the committed position plus the drafted
   ones. This is the only decoder pass in the round, and it replaces the `N` sequential passes the
   autoregressive loop would have made.
3. **Accept.** Walk the proposals in order, accepting each with probability `min(1, p/q)` where
   `p` is the verified distribution and `q` the draft's. On the first rejection, resample that
   position from the normalized residual `norm(p - q)+` and stop. If every proposal is accepted, a
   bonus token is sampled from the trailing verified distribution.

Because acceptance and residual resampling follow the standard rejection-sampling construction,
the committed codes are distributed exactly as the autoregressive loop would have produced. The
speedup comes from replacing sequential decoder passes with one wider pass, not from
approximating the model.

## Kernels

The sampling and verification kernels are specialised for the vocabulary size involved. A
residual-VQ codebook has a few thousand entries, small enough that one row fits in shared memory;
the general-purpose DSpark kernels select the top-k by rescanning the row once per selected
entry, so their cost grows with `top_k`. The CodePredictor path instead runs one MSB-radix select
followed by a bitonic sort over the surviving candidates — a single pass whose cost is
independent of `top_k`.

Semantics, tensor layouts, and accept/residual/bonus behaviour match the DSpark entry points
exactly; DSpark remains the fallback above the small-vocabulary bound.

## Limitations

- **Greedy decoding differs from the autoregressive path.** Verification needs a different
  `lm_head` per position, which the engine's single `lm_head_idx` gather cannot express, so the
  runtime applies the heads itself. The two computations round differently, and under greedy
  decoding that can change which token wins. Under sampling — the production configuration — the
  output distribution is unchanged.
- **Chain only.** Each round drafts a single linear sequence. A tree-shaped window would raise
  acceptance further but needs a tree verifier kernel with non-greedy sampling support, which is
  tracked separately.
- **Automatic fallback.** The runtime disables speculation, with a warning, when the codebook is
  larger than the small-vocabulary kernels support, when the CodePredictor engine predates the
  widened generation profile, or when the model has fewer than two RVQ depths. An engine built
  before this feature therefore keeps working, autoregressively.

## See also

- [Qwen3-Omni](../examples/omni.md) and [Qwen3-TTS](../examples/tts.md) for the full audio
  pipelines
- [Speculative Decoding](../examples/speculative-decoding.md) for the draft-model modes (MTP,
  EAGLE3, DFlash, DSpark, JetSpec)
