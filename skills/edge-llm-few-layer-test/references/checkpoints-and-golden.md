<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Choosing a checkpoint, and proving the golden is real

Both halves of this test can be wrong in ways that produce confident numbers. The
checkpoint can be a variant the builder cannot build, or one whose name misdescribes its
contents. The golden can be a randomly-initialised model that HuggingFace loaded without
complaint. Neither announces itself.

## Ask for the checkpoint

The right variant is often not the one named after the model. Three worked examples, each
a different way for the obvious choice to be wrong:

- **`Qwen3.6-35B-A3B`** — the plain checkpoint is FP16, and the checkpoint-direct
  builder's `qwen3_5_moe` expert op does not accept FP16 experts. The case has to use
  the `-NVFP4` variant. Picking by name gets a build failure that looks like a product
  bug.
- **`Qwen3.8-27B-NVFP4`** — the name says NVFP4; the recipe is mixed. The golden reports
  `NVFP4=13, FP8=13` and the builder logs `quant=fp8`. The FP8 half is on a far finer
  grid than NVFP4's E2M1, which is why it measures 0.999 where a quantized dense model
  is otherwise expected near 0.97. Had the threshold been set from the name rather than
  the log line, it would have been wrong in both directions.
- **A requested name can differ from a checkpoint on disk by a transposition rather than
  matching it** — `Nemotron3.5 Lightning 30B-A3B` against `Nemotron-Nano-3-30B-A3.5B`,
  where the `3` and the `3.5` swap places. Guessing builds a case that validates a
  different model than the one asked for, and it passes, so nothing ever surfaces the
  mistake.

Ask. Then confirm the builder accepts that variant before building a case around it.

## Prove the golden loaded real weights

`from_pretrained` does not fail on a checkpoint whose keys it does not recognise. It
reports them as unexpected, leaves the parameters at their random initialisation, and
runs.

`qwen3_asr` is the worked example: the checkpoint stores `thinker.model.layers.X`, while
transformers 5.14.1 expects `model.language_model.layers.X` for that class hierarchy.
Every key is unexpected, the golden runs on random weights, and the comparison produces a
cosine of approximately zero from round 0.

A run that is already broken in round 0 does **not** by itself implicate the golden. An
engine defect in an early layer lands there too: round 0 is the prefill, and it compares
every dumped layer, so a wrong weight, scale or kernel in layer 0 shows up immediately.

What separates the two is **which layer** diverges, not which round. Read the per-tensor
breakdown (`--verbose`) for round 0 and find the first tensor that disagrees:

- **Everything is wrong, including the earliest layer's state.** The golden never computed
  anything right. Suspect its weight loading first.
- **The earliest layers agree and the divergence starts partway down.** The golden loaded
  fine, and the first layer that disagrees is the suspect. A `nemotron_h` MoE case on SM120
  reads 1.00000 on the layer before the first MoE, 0.40 on the layer after it, and NaN
  after the second — an engine-side fault that round 0 reported immediately and that the
  layer-wise view localised.

Remember that the dumps use dense **cache-layer** numbering, which skips stateless layers
such as MoE and MLP, so a fault introduced by one of those appears on the next stateful
layer rather than on its own.

When the golden is the suspect:

- read its key report. For a quantized checkpoint the loader already fails loudly on
  missing keys (`quantized load: N missing keys`, ignoring `rotary_emb` and the
  legitimately-absent `lm_head.weight` of a tied embedding), and prints
  `unexpected selected keys=N`, which should be the layers truncation dropped and nothing
  else. A plain `from_pretrained` load is the quiet path — that is where `qwen3_asr`
  slipped through;
- check whether the whole run is uniformly dead rather than degrading with depth.

If the checkpoint's layout and the installed transformers disagree, few-layer validation
cannot be done for that model. Say so and ask the user what they want; a remap written to
get past the error produces a golden that is no longer a reference.

## Model-class resolution

The golden resolves a model class from the config's `model_type`. Vision-language configs
commonly register only under `AutoModelForImageTextToText` and not under
`AutoModelForCausalLM`; `_resolve_causal_lm_cls()` in `tests/golden_layer_dump.py` tries
`AutoModelForCausalLM` first and falls back to `AutoModelForImageTextToText`. Their
`forward()` still honours the CausalLM contract when no pixel values are passed, which is
what this text-only fixture needs.

A new family that registers under neither is a stop-and-ask, not a thing to force.

## Attention implementation

The golden asks for SDPA and falls back to eager only if the model does not implement
it, printing `[golden] SDPA unavailable (...); using eager attention` when it does. The
preference is not stylistic: SDPA accumulates QK^T in fp32, matching what the engine
does, while plain eager fp16 attention **overflows** for models without qk-norm whose K
can be large — Qwen2, with K around 130, drives QK^T past the fp16 maximum and produces
NaN.

So if that line appears in your log, the golden is on the degraded path, and NaNs or a
wildly low cosine may be the golden's arithmetic rather than the engine's. Read the line
before reading the numbers.

The two paths also do not produce bit-identical results where both work, so a floor
measured under one is not transferable to the other.

## Layer count is a choice, not a default

`--num-layers 4` is the default and usually right, but the window has to contain what the
case claims to protect.

`NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` uses six, because six lands on a `M E M E M *`
window and exercises Mamba, MoE and attention in one run; four would stop short of the
attention layer and quietly test less than the case says it does.

Check the golden's reported `layer_types` before settling on a number.

## Cache layers are not decoder layers

MLP and MoE layers carry no KV, conv or recurrent state, so the dense cache-layer
numbering both sides use is shorter than the decoder-layer count on any hybrid. The
driver derives the cache-layer count from the golden's `layer_types` metadata for this
reason. Passing a raw decoder-layer count where a cache-layer count is expected drops the
last layers' dumps without failing the run — fewer tensors compared, still green.

## Speculative drafts

A DFlash or EAGLE3 draft is a second, untruncated checkpoint: only the base is cut to
`--num-layers`. Declare it in the case as `spec_decode_config` plus `draft_id_or_dir`,
not as a path inside the inference `arguments` list. The declared form resolves through
the checkpoint resolver and lets `requirements.min_gpu_memory_gb` account for the draft;
a path in `arguments` is passed to the driver verbatim and does neither.
