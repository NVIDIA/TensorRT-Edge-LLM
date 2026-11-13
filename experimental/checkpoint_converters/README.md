<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Flax checkpoint converter

The converter transforms a supported local Orbax/Flax checkpoint into the
Hugging Face-style checkpoint directory consumed by TensorRT Edge-LLM. JAX and
Flax are used only while reading and transforming checkpoint tensors; the
resulting directory follows the existing Edge-LLM checkpoint interface.

## How it works

```text
CLI model-family selection
    -> family ConversionRequest validation
    -> family OrbaxCheckpointReader restore
    -> family CheckpointAdapter mapping and HF config
    -> shared tensor transforms
    -> shared HFCheckpointWriter with family write policy
    -> atomic HF-style checkpoint directory
```

The package root contains family-independent interfaces and implementations:

- `schema.py`: abstract request, common path/resource validation, errors, and
  conversion result.
- `converter.py`: abstract request construction and conversion interface.
- `source.py`: abstract Orbax reader and common restore error handling.
- `adapter.py` and `variant.py`: abstract mapping and model-variant interfaces.
- `transforms.py`: shared layout and dtype transformations.
- `writer.py`: deterministic safetensors partitioning, serialization, indexing,
  hashing, and atomic publication.
- `models/__init__.py`: private family-converter registry.
- `cli.py`: common argument parsing and abstract converter dispatch.

Each `models/<family>/` package owns its request constraints, source API,
adapter, variant definitions, tokenizer validation, provider assets,
provenance, and writer policy.

## Installation

Use an isolated Python 3.12 environment so converter-only JAX/Flax dependencies
do not become Edge-LLM runtime dependencies:

```bash
python3.12 -m venv .venv-flax-converter
source .venv-flax-converter/bin/activate
python -m pip install -r experimental/checkpoint_converters/requirements.txt
```

## Obtain Gemma4 inputs

The converter does not download checkpoints or tokenizer assets. Download all
objects under one of the supported public Orbax checkpoint prefixes and
preserve their relative paths in one local checkpoint directory:

- [Gemma4 E2B IT](https://storage.googleapis.com/gemma-data/checkpoints/gemma4-e2b-it/)
- [Gemma4 26B A4B IT](https://storage.googleapis.com/gemma-data/checkpoints/gemma4-26b-a4b-it/)

Obtain tokenizer assets from the matching pinned Hugging Face revision listed
in the support matrix.
The tokenizer directory must contain `generation_config.json`, `tokenizer.json`,
`tokenizer_config.json`, and `chat_template.jinja`. The converter preserves
these files byte-for-byte in its output.

## Interface

```bash
python -m experimental.checkpoint_converters.cli \
  --model-family gemma4 \
  --model-variant gemma-4-26B-A4B-it \
  --checkpoint-path /path/to/orbax-checkpoint \
  --tokenizer-path /path/to/tokenizer-assets \
  --output-path /path/to/hf-checkpoint
```

| Argument | Description |
| --- | --- |
| `--model-family` | Registered family converter to use. |
| `--model-variant` | Exact family variant. Defaults to `gemma-4-E2B-it`. |
| `--checkpoint-path` | Existing local Orbax checkpoint directory. |
| `--tokenizer-path` | Existing local tokenizer asset directory required by the selected family. |
| `--output-path` | Absent or empty destination directory. |
| `--restore-concurrent-gb` | Orbax restore concurrency bound in GB; default `16`. |
| `--max-checkpoint-file-bytes` | Planned tensor payload bound per safetensors file; default `5 GiB`. |

The converter validates all inputs before restoring model memory. Publication
uses a sibling staging directory and one final rename, so an incomplete
conversion is not exposed at the requested output path.

## Output

A successful conversion publishes:

- `model-*-of-*.safetensors`
- `model.safetensors.index.json`
- `config.json`
- family-required tokenizer assets, including the provider chat template
- `conversion.json` containing source/reference revisions, dtype policy,
  mapping digest, file hashes, and tensor counts

Correctness is validated through the normal Edge-LLM end-to-end sequence:
checkpoint conversion, export or direct build, engine build, and inference.

## Adding a model family

A new family package implements the shared interfaces and registers one
`CheckpointConverter` instance in `models/__init__.py`:

```text
models/<family>/schema.py   ConversionRequest subclass and validation
models/<family>/source.py   OrbaxCheckpointReader subclass
models/<family>/converter.py  CheckpointConverter implementation
models/<family>/flax_adapter.py  CheckpointAdapter implementation
models/<family>/variant.py  CheckpointVariant implementation
models/<family>/writer.py   CheckpointWritePolicy implementation
```

Family-specific tensor names, tokenizer rules, source APIs, revision pins, and
oversize tensor policies must not be added to the package-root modules.

## Support matrix

| Family | Variant | Source format | Source revision | HF reference revision | Conversion | Modality |
| --- | --- | --- | --- | --- | --- | --- |
| `gemma4` | `gemma-4-E2B-it` | Orbax, canonical FP32 | `google-deepmind/gemma@0f7e7420e84c6789d96ba465608d2a008ecb443d` | `google/gemma-4-e2b-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7` | FP32 to BF16, round-to-nearest-even | Text only |
| `gemma4` | `gemma-4-26B-A4B-it` | Orbax, canonical FP32 | `google-deepmind/gemma@0f7e7420e84c6789d96ba465608d2a008ecb443d` | `google/gemma-4-26B-A4B-it@4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | FP32 to BF16, round-to-nearest-even | Text only |
