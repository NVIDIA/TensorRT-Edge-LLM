# In-Flight Batching

In-flight batching lets the server and Python API serve several
requests at once. Requests share the running batch and a new request joins at
the next generation boundary rather than waiting for the batch to finish, so
concurrent clients see their first token while earlier requests are still
streaming. It is opt-in and, in this release, runs on a single device.

## Enable It

Server:

```bash
trtedge serve Qwen/Qwen3-8B-FP8 \
  --max-batch-size 4 \
  --enable-in-flight-batching \
  --port 8000
```

Python API:

```python
from tensorrt_edgellm.server import LLM

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
| Admission | Bounded queue with `--queue-timeout`; overflow is 429 | Limit of `max_batch_size` + `--max-queued-requests`; a request past it gets an immediate 429 (under `--scheduling-policy priority`, a request with priority below 0 may exceed it by `max_batch_size`) |
| Client disconnect | Cancels the native channel | Cancels the request inside the engine; its seat is freed |
| `/health` | `max_num_seqs: 1` | `in_flight_batching: true`, `max_num_seqs` = batch size, a `scheduling` block of engine counters and an `engine` block with the scheduling mode in effect; HTTP 503 once the engine no longer serves (see [Reading `/health`](#reading-health)) |
| Direct Python API | One call at a time | Calls from several threads overlap, up to the same batch-plus-queue limit; extra callers block |

Each request decodes with its own sampling parameters (temperature, top-k,
top-p, seed), `max_tokens` and logprobs setting, so requests that differ in
these join a running batch directly. A request whose thinking mode differs from
the running requests' waits until they have finished and is then admitted on its
own terms. Guided decoding is stricter: a guided request waits the same way but
then runs alone, and nothing joins it until it finishes. The `scheduling`
counters in `/health` show how often a request had to wait and why
(`stalls_incompatible`, `stalls_guided`, `stalls_no_capacity`).

## Scheduling Modes

How a new request's prompt shares the GPU with requests that are already
generating is set by `--ifb-scheduling-mode` (Python: `ifb_scheduling_mode`):

| Mode | Behavior |
|---|---|
| `mixed` | Running requests keep producing one token per step; a new prompt is processed in chunks inside the same steps, so a long prompt does not stall token streams, and several prompts can be processed together. |
| `prefillFirst` | While a new prompt is pending, the running requests wait and the prompt is processed (in one step when it fits the engine's maximum input length); then everyone decodes again. |
| `auto` (default) | `mixed` when the deployment supports it, else `prefillFirst`. The startup log names the mode chosen and why. |
| `singleCohort` | One new request is admitted per generation boundary and its whole prompt is processed alone while the running requests wait; then everyone decodes again. Compact execution is never used, so the outputs match in-flight batching without compact execution bit for bit. Use it to reproduce such outputs, to debug, or as a fallback. |

`mixed` needs kernels that run prefill and decode rows in one step (compact
execution). The runtime decides at startup, from the engine's layer kinds and
shapes and the GPU, whether every layer has them. The startup log
states the execution next to the mode, `execution=compact` or
`execution=single-cohort` (Python: `LLM.get_engine_status()["execution"]`),
because the same mode name can run either way (see below). Compact execution is
available for text and multimodal models whose layers support it on the GPU in
use (for example Qwen3 and Qwen3.5 on Blackwell data-center GPUs, and Qwen3.5 on
SM80 to SM90 GPUs such as Orin when the build's CuTe DSL artifact carries the
`gated_delta_net` and `causal_conv1d` kernels for that SM); image and
audio prompts are processed in chunks like text. It needs an engine built for
a batch size of at least 2. Deployments without it use `prefillFirst`, for
example TensorRT-RTX engines, n-gram embeddings and Gemma4
vision-bidirectional attention. Gemma4 per-layer embeddings stored with
`--ple-residency cached` need at least `min(PLE table rows, batch size x max
input length) + 1` `--ple-cache-slots`, because one compact step looks up the
rows of all its tokens at once; with fewer slots the deployment uses
`prefillFirst` without compact execution. The
[developer guide](../../developer_guide/software-design/in-flight-batching.md#execution-and-strategy)
lists every condition. Asking for `mixed` where it is not available fails at
startup with the reason. When a kind of layer has no such kernel on the GPU in
use, the reason names it with the GPUs that have one in the build in use. For
example, a build with the Mamba2 SSD kernels for Blackwell and for SM120/SM121
reports on SM87 `no compact-execution kernels on SM87 for Mamba2 SSD layers
(head dim 64, state size 128: no packed SSD prefill kernel for this SM, head dim
and state size in this build; supported on
SM100/SM101/SM103/SM110/SM120/SM121)`. An engine whose `config.json` lacks
`builder_config.compact_execution: true` runs without compact steps; rebuild it
with the current builder. The builder writes the field when every layer that
has a compact path reserved its compact workspace, and otherwise logs a warning
that names the layers without one. The server rebuilds its cached engines
automatically.

Compact execution belongs to the deployment, not to the mode. Where it is
available, `prefillFirst` uses it too: several waiting prompts are processed
together in prompt-only steps, and decode steps still replay the captured CUDA
graphs. Only `singleCohort` turns compact execution off on such a deployment. Guided requests,
media requests under a visual-token pruner and, with compact execution, Nemotron-Omni video
requests run alone, as described below.

Compact execution gives the same task accuracy as `singleCohort`, but not
bit-identical outputs: a step's kernels and floating-point rounding depend on
which requests share it. Under `mixed`, which requests share a step depends on
when they arrive, so repeating the same requests can produce slightly different
text near ties. Use `singleCohort` when you need outputs that are reproducible
bit for bit.

With vision or audio input, a deployment with compact execution reserves device
memory for the encoder outputs of every slot in every mode except `singleCohort`
(`auto`, `mixed` and `prefillFirst`), so
that a prompt's media rows stay available across its chunks. The reservation holds
FP16 values and takes

```text
B x (E_image x (1 + D) + E_audio) x H x 2 bytes
```

where B is the batch size the engine was built with, H the model's hidden size,
D the number of deepstack feature layers (3 for Qwen3-VL, 0 for most models),
`E_image` the smaller of the engine's maximum input length and the rows of the
vision encoder's output buffer (0 without a vision encoder), and `E_audio` the
engine's maximum input length (0 without an audio encoder). For example, Qwen3-VL-8B
(H = 4096, D = 3) built with batch size 4 and a maximum input length of 4096, with
an encoder buffer of at least 4096 rows, reserves 4 x 4096 x 4 x 4096 x 2 bytes =
512 MiB. The startup log reports the size (`Compact multimodal reservation`). If
the reservation cannot be allocated, the startup log warns with the reason and
size, `auto` serves the deployment without compact execution (as `prefillFirst`
on a deployment without it), and `mixed` fails at startup with that reason.
`singleCohort` never allocates it. Compact steps with prompt rows also run on a
second TensorRT execution context; if it cannot be created, the startup log
warns and compact execution stays available, with those steps running on the
original context.

The startup check is meant to keep a plugin from refusing a compact step at
runtime. If one refuses anyway, it logs an error that names the layer, and
compact execution stays off until the server restarts. What happens next
depends on the configured mode:

- `auto` or an explicit `prefillFirst`: the server keeps serving in
  `prefillFirst` mode without compact execution and without `--preemption
  pause`, and logs the new mode in a warning. A running request that has not
  produced a token yet goes back to the queue at its original place, ahead of
  later arrivals except, under `--scheduling-policy priority`, more urgent ones,
  and runs again from the start of its prompt; its client sees no error. A request that
  has already produced tokens fails with error type `compact_step_refused` (see
  [Errors](#errors)): HTTP 503, or on a stream, which has already started, an SSE
  error event of that type. It can be retried.
  `scheduling.compact_refusals` and the `engine` block of `/health` show the
  change. If the engine cannot serve again (no new runtime session opens, or
  the scheduling cannot be resolved again), it stops instead: these requests
  fail with `engine_unavailable`, as every later request does.
- An explicit `mixed`: the server stops serving. Every running request fails
  with an error that names the mode, every later request fails with the reason,
  and `/health` returns HTTP 503.

Budgets of a `mixed` step (0 selects the default). Batch size and max input
length here are the limits the engine is built with, `--max-batch-size` and
`--max-input-len` (the server reuses a cached engine only when it was built with
them). A budget outside its range stops the server when the engine starts, in
every mode, with an error that names the flag:

| Flag | Default | Range | Meaning |
|---|---|---|---|
| `--max-num-tokens` | about 512 + batch size, rounded up to 128 | batch size + 128 .. batch size x max input length | Tokens one step processes while requests are decoding |
| `--max-chunk-tokens` | up to 512: `--max-num-tokens` - batch size rounded down to 128, at most 512 | multiple of 128, up to `--max-num-tokens` - batch size | Prompt tokens of one request per step while requests are decoding |

One prompt chunk holds at most the max input length (rounded down to 128), so
a larger `--max-chunk-tokens`, or a larger default, is capped there; for an
explicit value the startup log says so in a warning. The startup log and
`/health` show the budgets in effect.

A prompt chunk that does not finish the prompt ends on a multiple of 128 tokens
and produces no output; the first token comes from the last chunk. With no
request decoding, a prompt runs in one step. When the mode resolves to
`prefillFirst` or `singleCohort`, an explicit `--max-num-tokens` or
`--max-chunk-tokens` is not used, and the startup log says so in a warning.

When the engine refuses a scheduling configuration at startup, the server exits
with `invalid server configuration:` and the reason, naming the options by their
command-line flags (for example `--ifb-scheduling-mode mixed is not supported by
this deployment: ...`). Combinations that do not depend on the engine, such as
`--preemption pause` with `--ifb-scheduling-mode prefillFirst`, are refused
before the engine loads.

### Priorities

`--scheduling-policy priority` orders waiting requests, and the prompt budget
of each step, by the request's `priority` field (lower is more urgent, default
0; the OpenAI-compatible `priority` field is forwarded). Requests of equal
priority keep arrival order. The order is strict and does not age: while more
urgent requests keep arriving, less urgent ones keep waiting. The server
refuses a request whose priority is below `--min-request-priority` with HTTP
400. The minimum is at most 0, so a request without a priority is always
accepted. The default 0 refuses urgent (negative) priorities; to accept them,
set the minimum to the most urgent priority your clients may send, for example
`--min-request-priority -1`. Only `/v1/chat/completions` and `/v1/completions`
carry a `priority`; `/v1/responses` and `/v1/messages` requests run at priority
0. These options build on the priority policy:

- `--preemption pause` (requires `--scheduling-policy priority`): while a more
  urgent request's prompt is processed, less urgent running requests are
  paused. They keep their state and resume with their next token once no
  prompt more urgent than theirs is unfinished, so a second urgent request
  that arrives in the meantime extends the pause. When no running request of the same or
  a more urgent priority is generating beside it, the urgent prompt runs in
  one step, so its time to first token approaches that of a request running
  alone; otherwise it shares each step with them and can take several. Pausing
  needs the `mixed` mode. Under the default
  `auto`, a deployment without compact execution resolves to `prefillFirst`
  and starts without pausing, with a warning: there, running requests already
  wait while a prompt is processed. With an explicit `prefillFirst` or
  `singleCohort`, the server refuses `--preemption pause` at startup.
- `--max-pause-steps N` (requires `--preemption pause`): a request paused for N
  steps in a row is not paused in the next step, then may be paused again. A
  request that is generating then produces its next token; one whose prompt is
  still being processed competes for that step's prompt budget in priority
  order and may still get no part of it. The default 0
  sets no limit: sustained urgent traffic keeps less urgent requests paused
  for as long as it lasts. Set a limit when such traffic is possible. A
  request that runs because of the limit generates beside the urgent prompt,
  which can then take more than one step.
- `--priority-reserved-slots N` (requires `--scheduling-policy priority`):
  requests with priority 0 or higher are admitted only while more than N slots
  are free, so an urgent request (priority below 0) finds a free slot even
  under load. HTTP clients reach these slots only with a negative
  `--min-request-priority`; with the default 0 the server warns at startup that
  no HTTP request can use them.

Under `prefillFirst`, a prompt-only step carries only the waiting prompts of the
most urgent priority; less urgent prompts wait for the next prompt-only step.

An urgent request starts at the next generation boundary, beside the running
requests, when all of these hold:

- A slot is free for it: a slot no request holds, or a reserved slot
  (`--priority-reserved-slots`) for a request with priority below 0. Running
  requests are never evicted to make room.
- Its sampling parameters match the running requests': temperature, top-p,
  top-k, the number of logprobs, `enable_thinking`, whether a sampling seed is
  set, `skip_special_tokens` and the context-cache policies. `max_tokens` may
  differ.
- It is not a request that runs alone (guided decoding, or media under a
  visual-token pruner), and no such request is running.

Without a slot, it waits for a running request to finish. With differing
sampling parameters, or beside a request that runs alone, it waits until every
running request has finished. Under `--scheduling-policy priority`, when such
a request is more urgent than every running request but waits for a parameter
mismatch, the server logs one warning for the request that names the parameter.
Running requests are never cancelled for it.

To serve urgent requests at once while all regular slots are busy, use an
engine built for k more slots than the regular load needs and reserve them. For
a regular load of 4 requests and one urgent request at a time (the engine is
built for a batch size of 5):

```bash
tensorrt-edgellm-serve Qwen/Qwen3-8B-FP8 \
  --max-batch-size 5 \
  --enable-in-flight-batching \
  --scheduling-policy priority \
  --priority-reserved-slots 1 \
  --preemption pause \
  --min-request-priority -1 \
  --port 8000
```

Regular requests then use at most 4 slots, and the engine pays the KV memory
of the reserved slot. With `--scheduling-policy priority` and no reserved slot
on an engine of two or more slots, the startup log warns that an urgent request
waits while all slots are busy. Under the priority policy the HTTP admission
limit also leaves room for urgent requests: a request with priority below 0 is
accepted up to `--max-batch-size` requests past the limit at which other
requests get 429, so it reaches the engine's priority queue under overload.

All of these flags except `--min-request-priority` require
`--enable-in-flight-batching`.

## Support Matrix

Deployments the flag cannot serve refuse it at startup with the reason in the
error message; they are never silently served one at a time.

| Deployment | Under `--enable-in-flight-batching` |
|---|---|
| Text LLM, single device | Served |
| Vision or audio **input** (VLM, ASR) | Served; media requests join the running requests like any other, and under `mixed` their prompts are chunked. Under a visual-token pruner a media request runs alone, and with compact execution so does a Nemotron-Omni video request |
| KV cache reuse (`--enable-context-reuse`) | Served; a joining request reuses a published prefix at admission |
| Guided decoding | Accepted, but never batched: a guided request waits for the running requests to finish, then runs alone, and no other request joins it while it runs |
| Audio output (Qwen3-Omni speech, standalone Qwen3-TTS) | Not available. An Omni bundle starts text-only: its speech engines are not loaded and `/v1/audio/speech` or audio output on chat is refused per request. A standalone TTS model refuses the flag at startup |
| Speculative decoding (MTP, EAGLE, DFlash, DSpark and the rest) | Served, with or without `--enable-context-reuse`; see [Speculative Decoding](#speculative-decoding) |
| Bounded sliding-window KV storage (`swaKVCacheMode=bounded`) | Refused at startup; use full KV storage |
| Tensor parallelism | Not available; the server and Python API run on a single device in this release |
| Hybrid Mamba models (Nemotron-H) | Starts and batches |

Requests that carry per-request LoRA or a trajectory are rejected with HTTP 400
under in-flight batching. Speech output is available on the default path only.

## Speculative Decoding

Speculative requests batch together. Two kinds of request wait for the running
speculative requests to finish instead of joining them:

- A request with `disable_spec_decode`. It runs on the default decoder, as on
  the blocking path, and batches with other such requests.
- A speculative request whose temperature, top-p, or top-k differ from the
  running speculative requests.

## Working With KV Cache Reuse

Both flags can be enabled together. A request that arrives while another is
decoding is admitted mid-flight and still reuses the published prefix, so its
time to first token stays close to a sequential reuse hit. Its prompt plus
`max_tokens` should fit `--max-kv-cache-capacity`; otherwise it is admitted with
its generation length clamped to what fits behind its own prompt, as on the
default path.

Each response's `usage.prompt_tokens_details.cached_tokens` reports how many
prompt tokens were served from reused state instead of being prefilled; on
streams it rides the usage chunk (`stream_options.include_usage`). The
`context_cache` block of `/health` carries the cumulative counters.

On a hybrid model such as Qwen3.5, a speculative request publishes only its
prompt: `cache_generated_tokens` defaults to `false`, and `true` is rejected
with HTTP 400.

## Reading `/health`

```json
"scheduling": {
  "submitted": 34, "completed": 33, "cancelled": 1, "failed": 0, "refused": 0,
  "admitted_mid_flight": 18,
  "stalls_incompatible": 152, "stalls_guided": 65, "stalls_exclusive": 0,
  "stalls_no_capacity": 0,
  "queue_latency_avg_us": 2669, "queue_latency_max_us": 42032,
  "decode_steps": 2210, "decode_graph_steps": 2210, "prompt_steps": 41, "mixed_steps": 30,
  "decode_subset_steps": 0, "step_sequences": 7921, "step_prompt_chunks": 52,
  "partial_prompt_chunks": 19, "step_decode_rows": 7869, "step_tokens": 31570,
  "step_sampled_rows": 7902, "step_time_total_us": 3605120, "step_time_max_us": 6210,
  "step_time_avg_us": 1601, "paused_resident_steps": 0, "compact_refusals": 0,
  "queued": 0, "resident": 0
},
"context_cache": {
  "admitted_sequences": 33, "hit_sequences": 21, "matched_tokens": 5504,
  "reused_tokens": 5376, "current_records": 12, "evicted_records": 0,
  "base_kv_pages": {"free": 40, "capacity": 96}, ...
},
"engine": {
  "serving": true, "unavailable_reason": null,
  "scheduling_mode": "mixed", "scheduling_mode_source": "auto",
  "scheduling_reason": "compact execution supported", "execution": "compact",
  "scheduling_policy": "priority", "preemption": "pause",
  "max_num_tokens": 640, "max_chunk_tokens": 512,
  "priority_reserved_slots": 1, "max_pause_steps": 0, "warnings": [],
  "compact_unavailable_reason": null
}
```

`admitted_mid_flight` counts requests admitted while other requests were
running, including a request admitted at the same step boundary as an earlier
one that has not run a step yet;
`stalls_*` count generation boundaries at which the queue head could not be
admitted and why; `queued` and `resident` are the live depths. `refused` is the
engine's
own back-pressure and stays at zero while the HTTP gate returns 429 first.

`context_cache` is filled when `--enable-context-reuse` is on and `null` otherwise. Its counters
are cumulative since startup and current as of the engine's last admission or
step; take differences between two reads to measure an interval.

The step counters cover every step that ran: `decode_steps` decoded every
running request (`decode_graph_steps` of them replayed a CUDA graph),
`prompt_steps` processed at least one prompt chunk (`mixed_steps` of them beside
decoding requests), `decode_subset_steps` decoded only some. `step_sequences`,
`step_prompt_chunks`, `step_decode_rows`, `step_tokens` and `step_sampled_rows`
are running sums; divide by the number of steps for the average step shape.
`step_time_*` is the scheduler's host time per step,
`paused_resident_steps` sums the requests paused by `--preemption pause` over
steps, and `compact_refusals` counts the compact steps a plugin refused.

The `engine` block is `null` without in-flight batching. `scheduling_mode` and
the scheduling fields after it are the scheduling in effect, which a refused
compact step can change after startup (see [Scheduling Modes](#scheduling-modes)):
`scheduling_mode_source` is `requested` when the configured mode is kept and
`auto` when `--ifb-scheduling-mode auto` chose it, in which case
`scheduling_reason` says why (it is `null` for a requested mode); `preemption` is the preemption
applied (`none` when `auto` dropped `pause`, and `max_pause_steps` is then 0), `max_num_tokens` and
`max_chunk_tokens` are the step budgets in effect (0 outside `mixed`), and
`warnings` lists the configuration the server accepted but does not apply as
given, as the startup log does. `compact_unavailable_reason` says why compact
execution is off. While the engine serves, `/health` returns HTTP 200 with
`status: "healthy"`. Once it no longer serves and fails every request, it
returns HTTP 503 with `status: "unhealthy"`, `engine.serving: false` and a
top-level `reason`. This happens after two session failures in a row with no
completed step between them (a failed step, or an admission that left the
device state unknown), when no new runtime session can be opened, when the
scheduling cannot be resolved again after a refused compact step, or after a
refused compact step under an explicit `mixed`; restart the server to
recover. A liveness probe on `/health` can therefore restart such a server.
`/health/ready` answers HTTP 200 with `status: "ready"` while the engine
serves, and HTTP 503 with `status: "unavailable"` and the same `reason` once it
does not, so a readiness probe takes the server out of rotation.

## Errors

Under in-flight batching, two failures return HTTP 503 instead of HTTP 500.
A client tells them apart by the error `type` in the OpenAI error body (`{"error":
{"type": ..., "message": ..., "code": 503}}`), or in the SSE error event of a
stream that has already started:

| `type` | When | `Retry-After` | What to do |
|---|---|---|---|
| `compact_step_refused` | A plugin refused a compact step after the request produced tokens; the server already serves again without compact steps | `1` | Retry the request |
| `engine_unavailable` | The engine no longer serves (see [Reading `/health`](#reading-health)); the message carries the reason | none | Restart the server; retrying does not help |

Any other engine failure stays HTTP 500 with type `engine_error`. On
`/v1/messages` the same statuses and `Retry-After` header apply, with the
Anthropic error body, whose type is `api_error` for both 503 responses; there
only `Retry-After` tells the retryable `compact_step_refused` failure apart.
