<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Test discipline

Applies to every test in this repository, C++ and Python alike.

## The Given/When/Then comment is the contract

Every test case carries a short Given/When/Then comment stating, in plain language, what behavior it
protects — not what the code does line by line. It exists so a reader can learn a test's intent without
reverse-engineering it from assertions.

Treat that comment as the specification and the test body as one rendering of it. **Passing is not the
goal; satisfying the stated behavior is.** The distinction only matters when the two come apart, which is
exactly when it matters most.

When a test fails, there are two legitimate responses:

1. Fix the production code so it satisfies the stated behavior.
2. Establish that the stated behavior is itself wrong.

The second is a human decision. An agent may argue for it; it may not enact it.

There is no third response. Loosening an assertion, narrowing the input, deleting one `EXPECT` of several,
or widening a tolerance — while the comment above still claims the original contract — is a silent removal
of coverage dressed as a fix. The comment stays and the contract goes, so the inconsistency is visible to a
reviewer who reads both. Do not create it.

Writing a new Given/When/Then, or editing an existing one, is likewise a human decision. Propose the
wording; do not commit it unreviewed.

## Mutation testing

A comment does not prove the assertion enforces the contract it describes. Before trusting a new assertion,
break the production code path it guards — flip a condition, skip a step, move a boundary — and confirm the
test goes red. Then revert.

This applies per assertion, not per file. A test can hold one real assertion and several that prove nothing.

### A green mutation has two explanations. Rule out the second first.

1. The assertion is a phantom — it looks like coverage and proves nothing.
2. **The mutation never took effect.**

The second is common and easy to miss:

- **You mutated a path nothing uses.** An accessor with one caller, or a single-sequence helper when the code
  under test takes the batch path. Grep for callers before concluding anything.
- **The mutation is vacuous for this fixture.** Forcing an index to zero proves nothing when the fixture has
  one sequence that already owns row zero. The fixture has to be able to tell the two apart.
- **The expectation moves with the mutation.** If the test compares against the same function that produced
  the value, mutating that function changes both sides and they stay equal. Give the test its own
  independently-stated expectation, and test the shared helper's non-degeneracy separately.

**Confirm the mutation changed the observed value** — print it if you have to — before concluding the test is
weak. A mutation that leaves the number identical has told you nothing about the test.

## Assert the work, not the accounting

Metrics, counters and status flags are usually computed on a different path from the work they describe. A
test that asserts only on them passes a system that reports a cache hit and then recomputes everything
anyway — precisely the bug worth catching.

Where a behavior has an observable consequence, assert the consequence: the sequence length a forward was
asked for, the tokens actually stored, the engine that did or did not run. Assert the counter as well if it
is part of the contract, but never only the counter.

## Prove the test can fail before trusting it

Several failure modes above share one root: a test that cannot distinguish the correct behavior from the
broken one, for reasons invisible in the test's own source. The mutation step exists to expose that. Budget
for it — it is not optional polish, it is the step that decides whether the rest of the work meant anything.
