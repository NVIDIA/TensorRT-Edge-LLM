---
name: edge-llm-serve
license: Apache-2.0
metadata:
  author: TensorRT Edge-LLM Team <noreply@nvidia.com>
  tags: [edge-llm, serve, openai-api, inference]
description: Launch and verify TensorRT Edge-LLM's experimental OpenAI-compatible server for one Hugging Face checkpoint or local checkpoint. Use for `trtedge serve`, OpenAI API requests, server sizing, streaming, tools, media, batching, speculative decoding, or a server startup failure. Do not use for ONNX export or C++ inference (edge-llm-convert-model), installation (edge-llm-install), or adding architecture support (edge-llm-add-model).
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Serve a checkpoint

Do the work; do not stop after printing a command. The default path is: launch,
wait until ready, send one request, report, stop the server.

## Instructions

Launch the server on loopback with a recorded PID and a log, wait for `/health/ready`, send one deterministic request that matches the user's modality, report the evidence, and stop the server unless the user asked to keep it. Add sizing, auth, media, tools, or batching only when asked.

## Source of truth

Read from the revision being run: `docs/source/user_guide/examples/server.md`,
`getting_started/support-matrix.md`, `getting_started/supported-models.md`, and
`docs/source/user_guide/features/in-flight-batching.md` for concurrency. Run
`trtedge serve --help`; it owns flag names, defaults, and value lists, and the
docs describe the checkout while `--help` describes the active installation.
Stop and report a mismatch instead of combining flags from both.

## Before launching

- The server consumes Hugging Face or local checkpoints, not ONNX or C++ engine
  directories. Do not switch to the ONNX workflow to work around a failure
  (`edge-llm-convert-model` owns that).
- Check the checkpoint against the support documents, and free device memory
  (`nvidia-smi --query-gpu=name,compute_cap,memory.free --format=csv,noheader`)
  against the checkpoint size.
- For a quick check without HTTP, `trtedge chat <checkpoint> --prompt "<text>"`
  generates once; it is not evidence that the server works.
- Environment: a published wheel with the `runtime` extra outside a checkout, or
  an editable source install backed by this checkout's build directory
  (`edge-llm-install`).

## Launch

1. Create a run directory and a cache directory for this task, and pick an unused port.
2. Start the server with the host's background-process mechanism, record the PID,
   and redirect output to a log. Do not use `pkill` or kill a process you did not start.

```bash
trtedge serve <checkpoint> \
  --host 127.0.0.1 --port <port> --cache-dir <task-cache> \
  --max-input-len <prompt-limit> --max-kv-cache-capacity <prompt-plus-output-limit>
```

3. The default host is `0.0.0.0`, so always pass `--host 127.0.0.1` unless the
   user asked for remote access (ask before exposing it).
4. The server listens only after the model loads, so a refused connection means
   "still loading" while the PID is alive. Poll `/health/ready` with a deadline
   and read the log; a cache-miss engine build can take minutes.

Size `--max-input-len` to the longest prompt and `--max-kv-cache-capacity` to
prompt plus generated tokens, and do not copy another checkpoint's profile. Take
every other option (batching, speculative decoding, tools and reasoning parsers,
`--api-key`, caching, queueing) from `--help` and only when the user needs it.
Three facts `--help` does not make obvious: speculative decoding cannot be
combined with in-flight batching; the default server serializes requests, so
concurrency needs `--enable-in-flight-batching` with a matching
`--max-batch-size`; and requests beyond the running batch plus
`--max-queued-requests` get HTTP 429. Use `--allowed-local-media-path` only for a
directory the user approved, and never print an API key.

## Prove the API

1. Save `/health` and `/v1/models`.
2. Send one request with `temperature: 0`, a bounded `max_tokens`, and the model
   name `/v1/models` returned. Match the modality: a known small image or audio
   input for media, `message.tool_calls` for tools.
3. For streaming, show at least one content or tool-call delta before the final event.
4. Record status, body, finish reason, usage, and the log line naming the engine
   bundle in use.

For latency or throughput use `edge-llm-benchmark-performance`; for dataset
accuracy `edge-llm-benchmark-accuracy`. Do not time one `curl`.

## Failures

| Failure | Action |
|---|---|
| Exits before ready | Report the first actionable log error (checkpoint access, memory, profile limits, native payload) |
| 400 / 404 / 413 | Request schema or unknown field / served model name / input over `--max-input-len` |
| 401 / 429 | API key / queue full |
| 503 | Shutting down, or the engine failed; read `/health` `reason` and the log |
| Plugin or TensorRT load error | `edge-llm-install`; do not reinstall over the environment blindly |
| Empty or wrong output | Keep the artifacts; `edge-llm-few-layer-test`, then `edge-llm-accuracy-debugging` |

## Report

The launch command with secrets redacted, checkpoint, port and PID, the request
and response excerpt, and whether the server was stopped or left running.

## Examples

- "Serve Qwen/Qwen3.5-4B on port 8000 and show that streaming works": size the limits, launch, wait for `/health/ready`, send a streamed request, report the evidence, stop.
- "Start the server for four concurrent clients": add `--enable-in-flight-batching` with a matching `--max-batch-size`, launch, and show that concurrent requests complete.
