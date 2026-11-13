---
name: edge-llm-benchmark-performance
license: Apache-2.0
metadata:
  author: TensorRT Edge-LLM Team <noreply@nvidia.com>
  tags: [edge-llm, benchmark, performance, aiperf, latency]
description: Benchmark TensorRT Edge-LLM serving performance (TTFT, inter-token latency, request latency, output throughput, concurrency scaling) through the OpenAI-compatible server with AIPerf. Use for "benchmark performance", "measure TTFT/ITL/tokens per second", prefill or decode sweeps over input and output lengths, in-flight-batching concurrency scaling, or comparing two deployments (quantization, speculative decoding, engine profile, another runtime). Do not use to launch a server only (edge-llm-serve), to score dataset accuracy (edge-llm-benchmark-accuracy), or to profile kernels.
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Benchmark serving performance with AIPerf

AIPerf measures what a client sees: HTTP, tokenization, queueing, prefill, and
decode together. Report it as served performance, not native kernel speed; use
the offline `llm_inference --dumpProfile` run for native numbers.

## Instructions

Take the workload lengths and concurrency from the user; then run Phases 1 to 5: prepare AIPerf, launch a sized server, choose the request shape, run, and validate with server token counts. Run Phase 6 only for a comparison. Report a table and the artifact paths.

## Read first

- `docs/source/user_guide/examples/integration/aiperf.md`
- `docs/source/user_guide/examples/server.md` (endpoints, request fields)
- `tensorrt_edgellm/server/api/protocol.py` (accepted request fields)
- `docs/source/user_guide/features/in-flight-batching.md` for concurrency tests

## Inputs

| Input | Required | Notes |
|---|---|---|
| Checkpoint and server launch | yes | Exact command from `edge-llm-serve`; the benchmark does not own the server |
| Workload | yes | Fixed prompts (JSONL) or synthetic input/output lengths; list the lengths |
| Concurrency | yes | 1 for latency; more only with in-flight batching |
| Comparison | optional | Second deployment to run back to back |

Ask when the lengths or concurrency are missing. Do not pick a "typical" load.

## Phase 1: environment

1. Install AIPerf in its own venv, never into the Edge-LLM runtime environment:
   `python3 -m venv .venv-aiperf && .venv-aiperf/bin/python -m pip install aiperf`.
   If pip fails resolving an unreachable extra index from a global pip config,
   add `--isolated --index-url https://pypi.org/simple`.
2. Record `aiperf --version` (call the venv's `bin/aiperf` by full path or put
   its `bin` on `PATH`). Compare results only between identical versions.
3. AIPerf needs a tokenizer matching the model. Pass `--tokenizer <id-or-dir>`;
   a directory must contain `tokenizer_config.json` and `tokenizer.json`. On an
   offline host use that local directory and do **not** set `HF_HUB_OFFLINE=1`
   (AIPerf 0.13.0 then rejects absolute paths). `--use-server-token-count`
   reports server token counts, but a tokenizer is still loaded.

## Phase 2: server for benchmarking

Launch with `edge-llm-serve` and pin everything that changes the work done:

- `--max-input-len` and `--max-kv-cache-capacity` cover the largest prompt plus
  output in the sweep. Keep them as small as the workload allows; do not raise
  them to the model maximum, which changes the engine profile being measured.
- The default server serializes requests (`--max-batch-size 1`). Concurrency
  above 1 measures queueing, not batching, unless the server was started with
  `--enable-in-flight-batching` and a `--max-batch-size` that matches. Record
  `--ifb-scheduling-mode` when set. Requests beyond the running batch plus
  `--max-queued-requests` (default 16) are refused with HTTP 429, so keep
  client concurrency within that sum.
- Leave `--enable-context-reuse` off unless reuse is the thing measured.
- Speculative decoding is a server property (`--speculative-config`) and
  combines with in-flight batching. Speculative requests batch only with
  requests that use the same temperature, top-p, and top-k, so keep sampling
  identical across a concurrency sweep.
- Warm the server with AIPerf's `--warmup-request-count` (Phase 4) after
  readiness; `/health/ready` only proves the engine loaded.

## Phase 3: choose the request shape

The server rejects unknown request fields with a 4xx error (`extra="forbid"`
on the request models), and the chat and completions endpoints accept
different fields. Use only these:

| Goal | Endpoint | Pin with `--extra-inputs` |
|---|---|---|
| Realistic chat latency, fixed prompts | `--endpoint-type chat --endpoint /v1/chat/completions` | `{"temperature":0,"top_k":1,"seed":42,"reuse_context":false}` |
| Exact input and output lengths (prefill/decode sweep) | `--endpoint-type completions --endpoint /v1/completions` | `{"ignore_eos":true,"temperature":0,"top_k":1,"reuse_context":false}` |

- `ignore_eos` exists only on completions. Sending it to chat fails every
  request. `seed` exists only on chat.
- Chat defaults are not greedy (temperature 0.7, top_p 0.9, top_k 50); always pin
  sampling (completions default to temperature 1.0 and top_k 0, also not
  greedy). Completions do not apply a chat template and chat adds template
  tokens; compare `usage_prompt_tokens` with the target, do not assume a match.
- Without `ignore_eos` a response can stop early and the measured output length
  will be shorter than requested; verify it in Phase 5.

## Phase 4: run

Fixed prompts (JSONL rows `{"text_input": "...", "output_length": 128}`):

```bash
.venv-aiperf/bin/aiperf profile --model <served-name> --tokenizer <tok> \
  --endpoint-type chat --endpoint /v1/chat/completions --streaming \
  --url http://127.0.0.1:<port> \
  --input-file prompts.jsonl --custom-dataset-type mooncake_trace \
  --concurrency 1 --request-count <rows> --warmup-request-count 2 \
  --extra-inputs '{"temperature":0,"top_k":1,"seed":42,"reuse_context":false}' \
  --output-artifact-dir <task-dir>/aiperf --ui none
```

Synthetic lengths, one run per length (completions endpoint from Phase 3):

```bash
.venv-aiperf/bin/aiperf profile --model <served-name> --tokenizer <tok> \
  --endpoint-type completions --endpoint /v1/completions --streaming \
  --url http://127.0.0.1:<port> \
  --synthetic-input-tokens-mean <isl> --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean <osl> --output-tokens-stddev 0 \
  --concurrency 1 --request-count 10 --warmup-request-count 2 \
  --extra-inputs '{"ignore_eos":true,"temperature":0,"top_k":1,"reuse_context":false}' \
  --output-artifact-dir <task-dir>/aiperf-isl<isl> --ui none
```

Use at least 10 measured requests per point and 2 warmup requests, more when the
run is short or the spread is large, and report the count.

Use `--streaming` so TTFT and ITL exist. Set `--request-count` explicitly; AIPerf
otherwise repeats a short file. Sweep input lengths (for example 128, 512,
2048) and concurrency (1, 2, 4 with matching batching) as separate runs against
one running server. Sample the device in parallel and keep the log:
`nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu -lms 250`.
Clocks and power drift between sessions, so never compare numbers across them.

## Phase 5: validate before reporting

Read `<dir>/profile_export_aiperf.json`. Each metric is an object with `unit`,
`avg`, and percentiles (`p50`, `p99`, ...); report `avg` and `p50` and say
which. Keys: `time_to_first_token`, `inter_token_latency`, `request_latency`,
`output_token_throughput_per_user`, `output_token_throughput`,
`request_count`, `completed_request_count`, `error_summary` (a list),
`request_error_rate`, `usage_prompt_tokens`, `usage_completion_tokens`
(server-reported), `input_sequence_length`, `output_sequence_length`
(client tokenizer), `osl_mismatch_count`.

1. `request_count` equals the requested `--request-count` and `error_summary` is empty.
   A 4xx means a refused field, 429 a full queue, 503 a failed engine; fix the
   cause, do not drop rows.
2. Trust the server's token counts. Check that `usage_completion_tokens`
   equals the requested output length and that one sampled response ends with
   `finish_reason: length` (`curl` it, or read `profile_export.jsonl`); then
   the run measured the intended work.
   AIPerf's `output_sequence_length` re-tokenizes the returned text and can
   differ, so an "output length mismatch" warning with correct server counts is
   a tokenizer artifact, not an EOS stop. If `usage_completion_tokens` is
   short, the model hit EOS: use the completions row or lengthen the prompts.
   For prefill work use `usage_prompt_tokens` (chat adds template tokens).
3. At concurrency 1, native decode rate is close to `1000 / ITL(ms)`.
4. Warnings about DCGM/GPU telemetry endpoints are harmless; use the
   `nvidia-smi` log instead.

## Phase 6: compare

Run deployments A and B interleaved (A, B, A, B) on the same device with
identical prompts, lengths, sampling, warmup, and server profile, and report
each run. If B is slower in any phase, report that as a loss with the numbers;
do not average it away. Speculative decoding: AIPerf reports latency only (no
acceptance rate); get acceptance from an offline `llm_inference --dumpProfile`
run with the same messages and limits, and do not infer it from ITL.

## Output format

Report in order: Edge-LLM revision, AIPerf version, platform and GPU
(clocks and power cap), checkpoint, redacted server command, AIPerf command per
run, a table with one row per input length and concurrency (target input
length and server `usage_prompt_tokens`, output length, TTFT, ITL, request
latency, output tokens/s per user and aggregate, completed/failed counts),
`nvidia-smi` summary, and artifact paths. If the server was started by someone
else, write "not launched by this run", give the settings readable from
`/health` and `/v1/models`, and list the rest as unverified.

## Limitations

- Client-side numbers include HTTP and tokenization; they are not kernel time.
- Server-side speculative counters are not exposed over HTTP.
- Single-session results on a shared or power-capped GPU vary by several
  percent; only interleaved A/B runs support a comparison.

## Troubleshooting

| Symptom | Cause and action |
|---|---|
| Every request 4xx | Unknown field (for example `ignore_eos` on chat); use the table in Phase 3 |
| `Failed to load tokenizer` | Offline or wrong path; see Phase 1 step 3 |
| Aggregate throughput flat as concurrency rises | Server not in-flight batching; relaunch per Phase 2 |
| 429 under load | Batch plus `--max-queued-requests` exceeded; lower concurrency or raise it |
| 503 on `/health` or requests | Engine failed; read the server log, relaunch |
| Output shorter than `output_length` | EOS hit; use completions with `ignore_eos` |

## Examples

- "Measure TTFT and ITL at input lengths 128, 512, and 2048 with 128 output tokens": use the completions endpoint with `ignore_eos`, one run per length at concurrency 1, validate `usage_completion_tokens`, report a table.
- "How does throughput scale from 1 to 4 clients?": launch with `--enable-in-flight-batching` and `--max-batch-size 4`, run concurrency 1, 2, and 4, and report aggregate and per-user tokens/s.
