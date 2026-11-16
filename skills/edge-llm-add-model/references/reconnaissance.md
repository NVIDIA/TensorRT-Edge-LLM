<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Checkpoint and implementation reconnaissance

Model onboarding starts with evidence, not a copied architecture name.

## 1. Inspect the exact checkpoint

Download only the checkpoint and revision the user named. Run:

```bash
python3.10 skills/edge-llm-add-model/scripts/inspect_checkpoint.py \
  <checkpoint-dir> --output <work-dir>/checkpoint-inventory.json
```

Use `python3.10` or newer (the script exits 2 with a message when the directory
has no safetensors shards, or when the index names missing shards or tensors, and
it reads top-level shards only). The script reads `config.json` and safetensors headers without importing
provider code. Review its root/nested `model_type`, architectures, markers,
sidecars, tensor patterns, missing shards, dtypes, and shapes.

For `.bin` weights, record that the script cannot inspect tensors and use a
separate isolated environment only with the user's approval. Never execute
checkpoint remote code as part of inventory.

Record the exact checkpoint commit. Reconcile the safetensors index and stop if
shards or indexed tensors are missing.

## 2. Inspect the provider reference

Use the `transformers` version pinned by this checkout. Locate the exact config
and modeling classes selected by the checkpoint. If `auto_map` points to remote
code, inspect that revision as source; run it only if the user approves.

Inventory:

- component topology and prefix names;
- decoder layer types and per-layer heterogeneity;
- attention, positional encoding, normalization, MoE/recurrent blocks;
- tensor names, fusions, transposes, packing, tied weights, and quantization;
- tokenizer, processor, chat template, media preprocessing;
- speculative draft or assistant checkpoints.

No PyTorch/provider reference means no trustworthy golden. Stop instead of
implementing from `config.json` alone.

## 3. Find the closest existing family

Do not rely only on a static file map. Search this revision for the nearest
family by topology, not model popularity, and read how its latest change was made:

```bash
git grep -n "<model_type>\|<architecture>" -- tensorrt_edgellm cpp docs
git log --oneline --no-merges -n 10 origin/main -- <closest-family-path>
git show <commit> -- <relevant-paths>
```

Adopt its patterns. Recent merged code is stronger evidence than an old path
listed in a skill.

## 4. State the plan

In a few lines: the checkpoint and revision, the `model_type` and components, how
the architecture and weights differ from the closest family, the route, and the
implementation class (alias, default reuse, family, component, operation).

Stop and ask when the checkpoint is ambiguous, incomplete, contradictory, uses
an unsupported quantization, needs an absent draft/adapter, or has no reference.
