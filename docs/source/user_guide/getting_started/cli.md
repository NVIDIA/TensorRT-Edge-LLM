<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

(edgellm-command-line-interface)=
# trtedge CLI

`trtedge` is the command-line interface (CLI) for chat, HTTP serving, engine
building, ONNX export, and model preparation.

```bash
trtedge COMMAND [OPTIONS]
```

(installation)=
Complete [Installation](installation.md) and activate your Python environment
before using `trtedge`. For a first run, follow [Quick Start](quick-start-guide.md).

## Flag spelling

Use hyphenated long options, such as `--max-input-len` and `--model-dir`.
The standalone C++ tools also accept their existing camelCase spellings
(for example, `llm_build --maxInputLen`), and Python model-preparation commands
retain underscore aliases (for example, `trtedge quantize llm --model_dir`).
Existing scripts can continue using those spellings. Configuration-file keys
and Python API parameter names are unchanged.

## Commands

The recommended installation includes `[server]` for chat and HTTP serving.
ONNX export requires `[export]`, and quantization requires `[tools]`.
See [Optional Python dependencies](installation.md#optional-python-dependencies)
to add these extras to your environment.

| Command | Purpose | Guide |
| --- | --- | --- |
| `trtedge chat` | Interactive or one-shot text chat | [Interactive Text Chat](../examples/interactive-chat.md) |
| `trtedge serve` | Launch an OpenAI-compatible HTTP server | [Python API and Server](../examples/experimental-server.md) |
| `trtedge build` | Build engines directly from a checkpoint | [Direct Engine Builder](direct-engine-builder.md) |
| `trtedge export` | Export checkpoint components to ONNX | {ref}`ONNX export example <export-the-checkpoint>` |
| `trtedge quantize` | Quantize an LLM or draft checkpoint | [Quantization](../features/quantization.md) |
| `trtedge insert-lora` | Insert LoRA patterns into ONNX | [LoRA](../features/lora.md) |
| `trtedge process-lora` | Process LoRA adapter weights | [LoRA](../features/lora.md) |
| `trtedge merge-lora` | Merge a LoRA adapter into a checkpoint | [LoRA](../features/lora.md) |
| `trtedge reduce-vocab` | Generate reduced-vocabulary artifacts | [Vocabulary Reduction](../features/reduce-vocab.md) |

## Examples

### Chat with a model

Start an interactive conversation with a Hugging Face model or a local checkpoint:

```bash
trtedge chat Qwen/Qwen3.5-0.8B
trtedge chat /path/to/checkpoint
```

To run one prompt:

```bash
trtedge chat Qwen/Qwen3.5-0.8B --prompt "Explain CUDA streams briefly."
```

Chat and serving build and cache TensorRT engines on first use. Subsequent
launches reuse matching cached engines.

### Launch a server

```bash
trtedge serve Qwen/Qwen3.5-0.8B --port 8000
```

See the [server guide](../examples/experimental-server.md) for HTTP requests,
speculative decoding, and runtime options.

### Build engines or export ONNX

Build TensorRT engines directly from a local checkpoint:

```bash
trtedge build --model-dir /path/to/checkpoint --engine-dir /path/to/engines
```

Export ONNX for the component-specific C++ builders:

```bash
trtedge export /path/to/checkpoint /path/to/onnx-output
```

### Quantize a checkpoint

```bash
trtedge quantize llm --model-dir /path/to/checkpoint \
  --output-dir /path/to/quantized --quantization fp8
```

Choose a quantization format supported by your model and target device; see
[Quantization](../features/quantization.md).

## Help and version

```bash
trtedge --help
trtedge chat --help
trtedge --version
```

Use `trtedge COMMAND --help` for command arguments and options.

## Standalone command names

You can also use the following standalone commands. Each accepts the same
arguments and options as its corresponding `trtedge` subcommand.

| `trtedge` command | Standalone command |
| --- | --- |
| `trtedge serve` | `tensorrt-edgellm-serve` |
| `trtedge build` | `tensorrt-edgellm-build` |
| `trtedge export` | `tensorrt-edgellm-export` |
| `trtedge quantize` | `tensorrt-edgellm-quantize` |
| `trtedge insert-lora` | `tensorrt-edgellm-insert-lora` |
| `trtedge process-lora` | `tensorrt-edgellm-process-lora` |
| `trtedge merge-lora` | `tensorrt-edgellm-merge-lora` |
| `trtedge reduce-vocab` | `tensorrt-edgellm-reduce-vocab` |
