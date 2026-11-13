<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Regression cases

Use these prompts against the current checkout.

## Plan a quantization fix

**Prompt:** Fix parsing for a ModelOpt checkpoint with INT4 AWQ weights and an
NVFP4 lm_head.

**Expected:** Before editing, list the touched axes: weight precision, lm_head
precision, checkpoint format, and both frontends. Find the counterpart parser
in the checkpoint-direct builder. Plan a checkpoint parse diff, a unit test for
the target checkpoint, and the cheapest evidence for each remaining row. Treat
the pair as supported: any backbone method combines with any lm_head method.

## Add a precision that partly exists

**Prompt:** Add support for an MXFP8 lm_head on NVFP4 weights.

**Expected:** Check existing support first. Find that the quantization CLI and
test precisions already accept an MXFP8 lm_head, and locate the gap per
checkpoint format and frontend. Start from the nearest existing test.

## Change a shared linear layer

**Prompt:** Add an optional input scale to the FP8 linear layer.

**Expected:** Find the callers one level deep, including drafts,
tensor-parallel variants, and the lm_head. Add rows for each caller and for the
checkpoint-direct builder. Give the exact command and hardware for each row
that cannot run locally.

## Trust a checkpoint name

**Prompt:** This checkpoint directory ends in `-FP8-KV`, so use it for the
FP8 KV rows.

**Expected:** Confirm the precision from the parsed quantization before using
the checkpoint. Report a mismatch between the name and the parsed result.

## Run only locally

**Prompt:** I only have an RTX GPU. Can I mark everything else as not run?

**Expected:** Run the rows the RTX GPU supports, and give the exact command and
hardware for each remaining row. Every touched axis still needs at least one
row with a log.

## Claim no interaction

**Prompt:** My change only touches the EAGLE3 draft. Do I need other rows?

**Expected:** Check the callers the change reaches. When the draft calls shared
base-model code, add the base rows that the shared code reaches. Otherwise,
list the untouched axes with the searches that support the claim.

## Add a speculative decoding method

**Prompt:** Add a new speculative decoding method based on the DFlash draft.

**Expected:** Search an existing sibling method name and each
`SpecDecodeMode::k*` switch, and add a row for every branch the method must
join or reject: reduced vocabulary, tensor parallelism, sampling, KV cache
reuse, hybrid recurrent bases, and the server method list. Check that each
inherited rejection names the new method. Add a model case for each
frontend.
