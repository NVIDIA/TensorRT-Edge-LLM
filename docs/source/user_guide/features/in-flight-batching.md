# In-Flight Batching

In-flight batching lets the experimental server and Python API serve several
requests at once. Requests share the running batch and a new request joins at
the next generation boundary rather than waiting for the batch to finish, so
concurrent clients see their first token while earlier requests are still
streaming. It is opt-in and, in this release, runs on a single device.

## Enable It

Server:

```bash
tensorrt-edgellm-serve Qwen/Qwen3-8B-FP8 \
  --max-batch-size 4 \
  --enable-in-flight-batching \
  --port 8000
```

Python API:

```python
from experimental.server import LLM

llm = LLM(model="Qwen/Qwen3-8B-FP8", max_batch_size=4,
          enable_in_flight_batching=True)
```

`--max-batch-size` (or `max_batch_size`) is the number of requests that decode
together; it must not exceed the batch size the engine was built with. Without
the flag both entry points serve one request at a time, which remains the
default.

## What Changes

| | Default (one at a time) | In-flight batching |
|---|---|---|
| Concurrent requests | Queued, served in order | Up to `max_batch_size` decode together; queued requests join at the next boundary |
| Admission | Bounded queue with `--queue-timeout`; overflow is 429 | Limit of `max_batch_size` + `--max-queued-requests`; a request past it gets an immediate 429 |
| Client disconnect | Cancels the native channel | Cancels the request inside the engine; its seat is freed |
| `/health` | `max_num_seqs: 1` | `in_flight_batching: true`, `max_num_seqs` = batch size, plus a `scheduling` block of engine counters |
| Direct Python API | One call at a time | Calls from several threads overlap, up to the same batch-plus-queue limit; extra callers block |

Sampling parameters that differ from the running batch (temperature, top-k,
top-p, `max_tokens`, penalties) do not fail. Such a request waits until the
current batch drains and then founds the next one, which later requests with
matching parameters can join. Guided decoding is stricter: a guided request
waits the same way but then runs alone. The `scheduling` counters in `/health`
show how often a request had to wait and why (`stalls_incompatible`,
`stalls_guided`, `stalls_no_capacity`).

## Support Matrix

Deployments the flag cannot serve refuse it at startup with the reason in the
error message; they are never silently served one at a time.

| Deployment | Under `--enable-in-flight-batching` |
|---|---|
| Text LLM, single device | Served |
| Vision or audio **input** (VLM, ASR) | Served; media requests join a running batch like any other. Under a visual-token pruner a media request founds its own batch |
| KV cache reuse (`--enable-context-reuse`) | Served; a joining request reuses a published prefix at admission |
| Guided decoding | Accepted, but never batched: a guided request waits for the running batch to drain, then runs in a batch of its own, and no other request joins it while it runs |
| Audio output (Qwen3-Omni speech, standalone Qwen3-TTS) | Not available. An Omni bundle starts text-only: its speech engines are not loaded and `/v1/audio/speech` or audio output on chat is refused per request. A standalone TTS model refuses the flag at startup |
| Speculative decoding (MTP, EAGLE, DFlash, DSpark) | Refused at startup |
| Tensor parallelism | Not available; the server and Python API run on a single device in this release |
| Hybrid Mamba models (Nemotron-H) | Starts and batches |

Requests that carry per-request LoRA, a saved system-prompt KV cache, or a
trajectory are rejected with HTTP 400 under in-flight batching. Speech output
is available on the default path only.

## Working With KV Cache Reuse

Both flags can be enabled together. A request that arrives while another is
decoding is admitted mid-flight and still reuses the published prefix, so its
time to first token stays close to a sequential reuse hit. Two constraints
apply to the joining request:

- `max_tokens` must match the running batch's; otherwise the request waits for
  its own batch (generation length is a batch-wide setting).
- Its prompt plus `max_tokens` must fit `--max-kv-cache-capacity`; otherwise it
  is refused for capacity until the batch turns over, and then founds its own
  batch with a clamped generation length.

## Reading `/health`

```json
"scheduling": {
  "submitted": 34, "completed": 33, "cancelled": 1, "failed": 0, "refused": 0,
  "admitted_mid_flight": 18,
  "stalls_incompatible": 152, "stalls_guided": 65, "stalls_no_capacity": 0,
  "queue_latency_avg_us": 2669, "queue_latency_max_us": 42032,
  "queued": 0, "resident": 0
}
```

`admitted_mid_flight` counts requests that joined a running batch;
`stalls_*` count generation boundaries at which the queue head could not join
and why; `queued` and `resident` are the live depths. `refused` is the engine's
own back-pressure and stays at zero while the HTTP gate returns 429 first.
