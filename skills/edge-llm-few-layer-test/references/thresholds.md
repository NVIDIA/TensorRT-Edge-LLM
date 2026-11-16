<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Setting a gate, and triaging a failure

A few-layer case is ambiguous in a way most tests are not. A red run may be a product
defect, or it may be the floor of a quantization recipe that no amount of correct code
will lift. The threshold is the only thing separating the two, which is why it has to be
derived rather than discovered by trying numbers until one passes.

The canonical reasoning for where each threshold comes from — why fp16 holds 0.9999, why
NVFP4 lands near 0.97, why MoE lands lower still — is in **"Where each case's cosine
threshold comes from"** in `docs/source/developer_guide/testing/few-layer-validation.md`.
Read it before choosing a number. This file covers what to *do*.

## Measured, not tuned

Two rules, and they pull in opposite directions on purpose.

**Do not set the gate to whatever the run produced.** `Qwen3.8-27B-NVFP4` measured
0.99892 and is gated at 0.96, the value its recipe uses elsewhere. A quantized threshold
expresses headroom across architectures and runners, not what one machine produced once.
A gate pinned to a single lucky measurement flakes the first time a tactic changes.

**Do not set the gate below the recipe's band to make a red run green.** That is the
silent removal of coverage described in the discipline reference: the case still claims
to protect the numerics, and no longer does.

Between them: pick the value the recipe already uses, confirm the measurement clears it
with room, and if it does not, triage before reaching for the threshold.

## Triaging a failure

Work through these before concluding anything. Each is cheap relative to the cost of
either chasing a normal quantization error for days or shipping a loosened gate.

**1. Which tensor class failed?**

Read the per-round lines, not just the summary. Logits and cache tensors fail for
different reasons. Logits sit after the final norm, so their magnitude depends only on
the hidden state's direction and they move far more than a cache tensor does. A failure
confined to logits while KV and recurrent state stay above 0.99 is the signature of
discrete disagreement — expert routing, most often — rather than arithmetic drift. A
failure that includes the cache tensors is much more likely to be real.

**2. Does it reproduce?**

Re-run unchanged. Neither side is bit-reproducible: TensorRT re-picks tactics per build,
and the reference MoE sums experts with an atomic `index_add`. A systematic defect
reproduces to the digit; a routing-driven failure moves.

Do not treat movement alone as proof of innocence, or stability as proof of guilt.
Several correct cases reproduce to five decimal places across separate builds. The signal
is whether the variation matches the mechanism you are proposing.

**3. Is it activation quantization?**

Re-run with `--no-quantize-activations`, which exports both sides with the activation
Q-DQ dropped. If the number recovers to ~0.999, activation quantization is the source and
the case is healthy. If it recovers only partway, the remainder sits on the weight grid
and no threshold change will address it. Diagnostic only — the engine is much slower.

**4. Take more than one sample before writing a number down.**

A single measurement cannot distinguish a floor from a draw. Where variation is real,
five runs cost less than one wrong threshold.

## A worked example

`Qwen3.6-35B-A3B-NVFP4` first ran 0.93892 against a 0.95 gate. The triage:

| Step | Result | Reading |
|---|---|---|
| Tensor class | all failures on logits; KV steady at 0.991 | discrete, not drift |
| Re-run | 0.92601 | not reproducible, so not systematic |
| `--no-quantize-activations` | 0.96489 | only partial recovery — residue on the NVFP4 weight grid |
| Six samples | 0.926, 0.929, 0.931, 0.938, 0.939, 0.948 | all on the logits of the same decode round |

The location was fixed and the value was not: one token's hidden state sits on an
expert-routing boundary, so each side legitimately selects a different expert set. The
other 127 rounds held a median of 0.991.

The sixth sample arrived later than the first five and landed outside their range, in the
healthy direction, widening the spread from 0.013 to 0.022. That is the argument for
sampling in step 4 rather than an argument against it: five draws of a discrete process
understate its tail, and a gate chosen to sit just under the first five would now be
inside the spread.

It is registered at 0.90 with that measurement recorded beside it. Note what the number
cost: a 0.90 gate still catches a dropped scale factor, a wrong kernel or a structurally
broken layer, and will not catch a 2% drift. **Say so when you set one.** A loose gate
that nobody has characterised is worse than a loose gate that has been.

The whole example was measured on a 128-round fixture, and the case has since moved to
the four-request ragged default. **A gate is only valid for the fixture it was measured
on**: changing the fixture invalidates the measurement the way changing the quantization
recipe would, so re-measure rather than carrying the number across.

Re-measuring here is also a caution against the obvious prediction. Six rounds instead of
128 is far fewer draws from the routing-boundary distribution, so the outlier that bought
0.90 looked likely to disappear. It did not — the six-round fixture measured 0.93781,
inside the original 0.926-0.948 spread. A disagreement that survives a twentyfold cut in
sampling was never a rare draw; it is what the recipe does, and no fixture change will
tidy it away.

## Thresholds outside the usual band

When a case needs a value outside the band its recipe normally uses, record next to the
case: the samples, which fixture produced them, where the worst value landed, the triage
that ruled out a defect, and what the gate no longer catches. A future reader otherwise cannot tell a characterised
exception from an unexplained one, and the safe assumption — that it was tuned until it
passed — is the damaging one.

## The known limitation

The gate is a **minimum** over every tensor, sequence and round, so one pathological
token sets it for the whole run. For quantized MoE this is routine: the healthy signal
sits at 0.991 and a single routing-boundary token drags the gate down by six points.

The obvious escapes do not work. More layers does not dilute a minimum, and adds routing
flips. A mean averages away the single broken layer this test exists to catch. A median
hides a fault confined to one tensor, which is the same objection.

This is tracked as its own issue; the direction under discussion is gating per tensor
class, so the state tensors that hold 0.991 can carry a strict threshold while logits
carry a loose one. Until then: a low MoE gate is a known limitation of the instrument,
not evidence that you configured the case wrongly.

## Mutation-test the gate

A threshold is an assertion, and an assertion nobody has seen fail proves nothing. This
matters most exactly where the gate is loose.

Perturb the numerics the case is meant to protect — a weight scale, a kernel selection —
confirm the case goes red, and revert. Then confirm the mutation actually changed the
measured cosine: a mutation that leaves the number identical has told you nothing, which
is the second explanation for a green mutation in the discipline reference.
