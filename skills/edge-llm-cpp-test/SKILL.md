---
name: edge-llm-cpp-test
license: Apache-2.0
metadata:
  author: TensorRT Edge-LLM Team <noreply@nvidia.com>
description: Write or change a C++ test under unittests/. Covers the BDD comment contract, mutation testing, and the substitute engines that drive LLMInferenceRuntime without a serialized engine. Apply whenever adding, migrating, or repairing a GTest case in this repository.
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# EdgeLLM C++ Tests

C++ tests live in `unittests/`. Build them with `-DBUILD_UNIT_TESTS=ON`; they need a GPU but no model files.

## Non-negotiable

1. **Every test case carries a Given/When/Then comment.** That comment is the specification. The test body is one rendering of it.
2. **Every new assertion is mutation-tested**, and the mutation is confirmed to have changed the observed value before any conclusion is drawn from it.
3. **Writing or changing a Given/When/Then comment is a human decision.** Propose it; do not decide it.

The reasoning behind all three, and the failure modes they exist to prevent, is in `references/test-discipline.md`. Read it before writing a test.

## Workflow

1. State the behavior you intend to protect as Given/When/Then, in plain language, before writing any assertion. If you cannot state it without describing the implementation, you do not yet know what the test is for.
2. Render assertions from that statement. Prefer asserting the observable consequence over the bookkeeping that reports it — see "Assert the work, not the accounting" in the discipline reference.
3. For runtime tests, assemble `LLMInferenceRuntime` around a substitute engine rather than a serialized engine. Which substitute, and how to observe through it, is in `references/runtime-test-infra.md`.
   For EAGLE or MTP draft substitutes, use `FakeEngineExecutor::actAsDraftEngine()` or `makeLogProbDraftEngine()`: draft logits must be log-probabilities, unlike a base engine's raw logits.
4. Build and run:
   ```
   cmake --build build --target unitTestRuntime -j$(nproc)
   export LD_LIBRARY_PATH=$TRT_PACKAGE_DIR/lib:$LD_LIBRARY_PATH
   ./build/unittests/unitTestRuntime --gtest_filter="YourSuite.*"
   ```
5. Mutation-test each new assertion, then revert the mutation and re-run the whole target to confirm the tree is clean.
6. Run the full group before reporting, not just your filter: a substitute or fixture shared with other suites can break them.

## When a test fails

Only two responses are legitimate: fix the production code so it satisfies the Given/When/Then, or establish that the Given/When/Then itself is wrong — which needs a human. Re-rendering the test so it passes while the comment stays put is not a third option.

## References

| File | Read when |
|------|-----------|
| `references/test-discipline.md` | Before writing or changing any test |
| `references/runtime-test-infra.md` | Writing a test that drives `LLMInferenceRuntime` |
