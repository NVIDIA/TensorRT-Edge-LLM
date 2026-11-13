<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmark the OpenAI Server with AIPerf

[AIPerf](https://github.com/ai-dynamo/aiperf) can benchmark the TensorRT
Edge-LLM OpenAI-compatible chat endpoint without an adapter. Use streaming
requests so it can measure time to first token (TTFT), inter-token latency
(ITL), request latency, and output throughput.

## Install and start the server

Install TensorRT Edge-LLM with the `runtime` extra as described in
[Installation](../../getting_started/installation.md#recommended-python-api-and-server),
then install AIPerf in a separate virtual environment:

```bash
python3 -m venv .venv-aiperf
.venv-aiperf/bin/python -m pip install aiperf
```

Keep this server short. AIPerf traces are written by you, so limit each prompt
to 8,192 tokens and each prompt plus `output_length` to 10,240 tokens instead
of raising the engine limits.

```bash
edgellm serve Qwen/Qwen3.5-4B \
  --max-input-len 8192 \
  --max-kv-cache-capacity 10240 \
  --port 8000
```

The model is downloaded on the first launch and the compiled engine bundle is
reused on later launches. Omit `--cache-dir` to use the default cache. Pass a
local checkpoint directory instead of the model ID to use an existing
checkpoint.

Wait until the readiness endpoint succeeds:

```bash
curl --fail http://127.0.0.1:8000/health/ready
```

## Run a reproducible single-request benchmark

Create a JSONL trace. Each row is one request, and this guide supplies the
prompt text, so AIPerf does not synthesize a fixed-length input. `output_length`
is sent as `max_tokens`. The completion reaches that length only when the model
stops on the limit:

```json
{"text_input":"Explain paged KV caches.","output_length":128}
{"text_input":"Compare static and continuous batching.","output_length":128}
```

Run AIPerf at concurrency 1 before testing server batching:

```bash
.venv-aiperf/bin/aiperf profile \
  --model Qwen/Qwen3.5-4B \
  --endpoint-type chat \
  --endpoint /v1/chat/completions \
  --streaming \
  --url http://127.0.0.1:8000 \
  --input-file prompts.jsonl \
  --custom-dataset-type mooncake_trace \
  --concurrency 1 \
  --request-count 2 \
  --warmup-request-count 2 \
  --extra-inputs '{"temperature":0,"top_k":1,"seed":42,"reuse_context":false}' \
  --output-artifact-dir artifacts/aiperf \
  --ui none
```

Set `--request-count` to the number of JSONL rows. Without it, AIPerf can
repeat a short file until it reaches its default request count. `--ui none`
prints the report without the live dashboard. Keep the prompt file, output
limits, sampling options, and warmup count fixed when comparing deployments.
`reuse_context:false` prevents repeated prompts or shared prefixes from
changing the amount of prefill work between runs.

## Example output

The table below is one example report. It comes from a single AIPerf 0.13.0 run
against `Qwen/Qwen3.5-4B` at concurrency 1, and it is not a target to match.
The trace above was sent once per row after two warmup requests. The server
used a cached engine with a 1,024-token input limit and 2,048-token KV
capacity.

| Metric | Average |
|---|---:|
| Time to first token | 26.47 ms |
| Inter-token latency | 15.82 ms |
| Request latency | 2035.56 ms |
| Output throughput per user | 63.21 tokens/s |
| Aggregate output throughput | 62.82 tokens/s |
| Server prompt tokens | 18.5 |
| Server completion tokens | 128 |
| Completed requests | 2 |
| Request errors | 0 |

AIPerf's own input count averaged 6.5 tokens. The server count is higher
because the chat template adds special tokens. Use `usage.prompt_tokens` when
comparing prefill work. The payload below is a separate short request against
the same server, shown only as an example of the response shape. Its prompt is
the text from the request, not a prompt AIPerf generated, and generation
stopped because it reached that request's `max_tokens`:

```json
{
  "object": "chat.completion",
  "model": "Qwen/Qwen3.5-4B",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Paged KV caches are a memory management technique that dynamically allocates and manages the key-value pairs used in transformer models by dividing them into fixed-size pages, allowing"
      },
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 22,
    "completion_tokens": 32,
    "total_tokens": 54
  }
}
```

The default server serializes requests. For a load test with actual concurrent
execution, restart it with:

```bash
edgellm serve Qwen/Qwen3.5-4B \
  --max-input-len 8192 \
  --max-kv-cache-capacity 10240 \
  --enable-in-flight-batching \
  --max-batch-size 4
```

Then repeat AIPerf with `--concurrency 2` and `--concurrency 4`. Do not interpret
concurrency greater than one as an in-flight batching result unless the server
was launched with `--enable-in-flight-batching`.

## Compare normal and speculative decoding

Use the same target checkpoint for both runs. For example, Qwen3.5-4B contains
an MTP draft layer:

```bash
# Normal decoding
edgellm serve Qwen/Qwen3.5-4B \
  --max-input-len 8192 \
  --max-kv-cache-capacity 10240

# MTP: three draft tokens and four target verification positions
edgellm serve Qwen/Qwen3.5-4B \
  --max-input-len 8192 \
  --max-kv-cache-capacity 10240 \
  --speculative-config \
  '{"method":"mtp","num_speculative_tokens":3}'
```

Run the concurrency-1 AIPerf command against each deployment. Compare:

- TTFT, which includes request handling, tokenization, prefill, first-token
  decode, and HTTP delivery.
- ITL and output throughput per user, which are the most useful client-side
  decode measurements at concurrency 1.
- Request latency and aggregate output throughput.

For EAGLE3, DFlash, DFlash2, dSpark, and JetSpec launch configurations, see
[Speculative Decoding](../speculative-decoding.md).

## Compare with offline inference

Build and run the same target and speculative configuration with
`llm_inference`, following the corresponding section of
[Speculative Decoding](../speculative-decoding.md). Add profiling:

```bash
./build/bin/llm_inference \
  --engineDir "$ENGINE_DIR" \
  --checkpointDir "$CHECKPOINT_DIR" \
  --inputFile native-input.json \
  --outputFile offline-output.json \
  --warmup 2 \
  --dumpProfile \
  --profileOutputFile offline-profile.json
```

For speculative decoding, add the mechanism-specific arguments. For the MTP
example above:

```bash
  --specDecode \
  --specDraftTopK 1 \
  --specDraftStep 3 \
  --specVerifySize 4
```

Use the same messages, output limit, batch size 1, and sampling values as the
AIPerf trace. Check that both paths report the same prompt and completion token
counts before comparing rates.

The metrics are related, but not interchangeable:

- Offline prefill throughput measures native model work. AIPerf TTFT also
  includes CPU, queueing, tokenization, first-token decode, and HTTP overhead.
- At concurrency 1, native decode tokens/s should be close to
  `1000 / AIPerf ITL (ms)`. Small differences come from token accounting and
  streaming overhead.
- AIPerf's client-derived active prefill throughput divides its tokenizer's
  input count by observed TTFT. Prefer the server `usage.prompt_tokens` count
  when it differs because chat-template and special tokens are added server
  side.
- Native speculative profiles report verification iterations, acceptance
  length, draft acceptance rate, and speculative throughput. A speedup without
  healthy acceptance is workload-specific and may disappear with different
  prompts or concurrency.

TensorRT Edge-LLM currently returns OpenAI `usage` fields but does not attach
native speculative counters to the HTTP response. Consequently AIPerf reports
latency and throughput for speculative deployments, but its Spec Decode table
is absent. Use `offline-profile.json` for acceptance analysis; do not infer
acceptance rate from ITL alone.
