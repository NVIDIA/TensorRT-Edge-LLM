<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Interactive Text Chat

Complete [Installation](../getting_started/installation.md) and activate your
Python environment. For a first run, follow [Quick Start](../getting_started/quick-start-guide.md#option-1-start-a-chat-directly).

Start a chat with a Hugging Face model ID:

```bash
trtedge chat Qwen/Qwen3.5-0.8B
```

Or use a local checkpoint directory:

```bash
trtedge chat /path/to/checkpoint
```

Follow build and loading progress in the terminal. Check the model, device,
checkpoint precision, and speculative decoding mode before the first prompt.

> **TensorRT engine caching**
>
> On initial launch, `trtedge chat` builds and caches the required TensorRT engines.
> Subsequent launches reuse matching cached builds. Changing TensorRT build
> settings triggers a rebuild unless a matching cached build already exists.
> Downloads and engines are stored in `~/.cache/tensorrt-edgellm` by default.
> Use `--cache-dir /path/to/cache` to choose another writable directory.
> Pass `--clear-engine-cache` at startup to clear cached TensorRT engines and
> force a clean rebuild.

Use `trtedge chat --help` for launch options. See the
[trtedge CLI](../getting_started/cli.md) for the full
command list and package options.

## Chat commands

Type a prompt and press Enter to send it. Alt+Enter adds a new line. Use the arrow
keys to edit or recall prompts, and Tab to complete commands or paths after
`/file`. Prompt history lasts for the current session.

| Command | Action |
| --- | --- |
| `/help` | Show commands and shortcuts |
| `/clear` | Reset the conversation to its initial messages |
| `/regen` | Replace the last response using the same prompt and settings |
| `/file PATH` | Send a UTF-8 text file as a prompt; spaces in paths need no quotes |
| `/exit` | Exit |
| Ctrl+C during generation | Stop the response |
| Ctrl+C at the prompt, or Ctrl+D | Exit |

Stopping a response preserves its displayed text in conversation history.
Use `/clear` to reset the conversation to its initial messages.

## Common options

| Flag | Purpose | Note |
| --- | --- | --- |
| `--max-tokens N` | Limit response length | Includes both thinking and answer tokens |
| `--enable-thinking` | Enable thinking output | Requires a model that supports thinking |
| `--system "MSG"` | Set a system prompt | Use `--system-file PATH` to read it from a file |
| `--max-input-len N` | Set the prompt limit, including history | TensorRT engine build option |
| `--max-kv-cache-capacity N` | Set total context capacity | TensorRT engine build option; must fit prompt and response |

Changing `--max-input-len` or `--max-kv-cache-capacity` triggers a TensorRT
engine rebuild on the next launch unless a build with the same settings is
already cached.

Leave room in the context for `--max-tokens`. To use quantized weights, supply a
checkpoint in a supported quantization format.

## Speculative decoding

See the [speculative decoding reference](experimental-server.md#speculative-decoding)
for supported methods and `--speculative-config` field definitions.

To use speculative decoding, choose a method supported by your checkpoint.
For [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), a 4B checkpoint
with native MTP weights, run:

```bash
trtedge chat Qwen/Qwen3.5-4B \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

This requests three draft tokens per draft verification step. For methods with
a separate draft model, such as EAGLE3, provide a compatible draft checkpoint:

```bash
trtedge chat /path/to/target-checkpoint \
  --speculative-config '{"method":"eagle3","model":"/path/to/draft-checkpoint","num_speculative_tokens":3}'
```

Pass `--disable-spec-decode` with the same command to compare generation with
speculation disabled. Check the `speculative mode` field at startup for the selected
mode.

## Run one prompt

```bash
trtedge chat /path/to/checkpoint --prompt "Explain CUDA streams briefly."
trtedge chat /path/to/checkpoint --prompt-file question.txt > answer.txt
```

- `stdout`: Generated response text for piping or redirection.
- `stderr`: TensorRT engine status, timing summaries, and diagnostic logs.

Pass `--no-color` to disable ANSI colors, `--no-metrics` to suppress timing
summaries, or `--verbose` to enable debug logs. Use `--system-file system.txt`
to read a system message from a file.

## Save settings

Save repeated options in `settings.json`. For Qwen3.5-4B MTP, use:

```json
{
  "speculative_config": {
    "method": "mtp",
    "num_speculative_tokens": 3
  },
  "max_tokens": 512
}
```

```bash
trtedge chat Qwen/Qwen3.5-4B --config settings.json
```

> **Configuration precedence**
>
> Highest to lowest priority:
>
> CLI flags → environment variables → settings file → API defaults.
>
> For example, `--max-tokens 512` overrides `EDGELLM_MAX_TOKENS=256`.

- **JSON keys:** Replace flag-name hyphens with underscores, such as
  `--max-tokens` → `"max_tokens"`.
- **Environment variables:** Use uppercase names with underscores and the
  `EDGELLM_` prefix, such as `--max-tokens` → `EDGELLM_MAX_TOKENS=256`.

Select a settings file with `--config` or `EDGELLM_CONFIG`.

## Timings

Chat shows a summary after each response. For example:

```text
prompt 603.2 tok/s | generation 4.8 tok/s | context 331/8.19K (4%) | TTFT 0.262s | total 36.915s
```

The summary covers one response and excludes model download, engine building,
and model loading.

| Metric | Meaning |
| --- | --- |
| `prompt` | Prompt processing speed in tokens per second. Excludes prompt tokens reused from the cache. |
| `generation` | Token generation speed after prompt processing, in tokens per second. Includes accepted speculative tokens. |
| `context` | Tokens used / context capacity (percentage used). Includes the full prompt, conversation history, and all output tokens. |
| `TTFT` | Time to first token: from starting the request stream to receiving the first token. |
| `total` | Time until the request stream ends, including output display and cleanup. |

Processing speeds use runtime measurements. `TTFT` and `total` also include
application overhead. In the context count, `K` means 1,000 tokens; displayed
counts are rounded.

Unavailable values display `n/a`. Stopped responses show the available metrics.
Use `--no-metrics` to hide the summary. For API details, see
[Streaming Metrics](experimental-server.md#streaming-metrics).
