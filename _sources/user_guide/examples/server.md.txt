# Python API and OpenAI-Compatible Server

The Python server accepts a supported checkpoint, builds all of
its runtime components directly, and exposes offline generation plus OpenAI and
Anthropic HTTP APIs. ONNX is not used by this path.

The supported API contracts and launch options are documented below. Optional
features may have model or platform limitations linked from their sections.

## Install

For a supported target, follow [published-wheel installation](../getting_started/installation.md#published-python-wheel)
and select `tensorrt-edgellm[runtime]==<version>`. The wheel includes the runtime
bindings, builder, and plugin, so `[builder]` is only needed for source builds;
no checkout or CMake build is needed. Install the TensorRT Python bindings
supported by the target platform before launching the server. Run the examples
in that environment outside a source checkout.

Alternatively, complete the [source build and server setup](../getting_started/installation.md#install-and-launch-the-python-server).
Do not reinstall the source package over a published wheel.

For gated checkpoints, accept the provider's terms and run `hf auth login`.
The runtime extra supplies `huggingface-hub` for downloads. Model-provided Jinja
templates run natively through Pantor Inja; serving does not require Jinja2 or
the PyTorch/ONNX exporter.

## Python API

`model` is a Hugging Face model ID or a local checkpoint directory. On a cache
miss, one builder invocation compiles every component owned by the model
family. Supported weights, including FP16 weights, are externalized by default
and loaded from the resolved checkpoint during runtime initialization. Weights
that the model family cannot externalize remain in the engine.

```python
from tensorrt_edgellm.server import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3.5-0.8B",
    cache_dir="/data/edgellm-cache",
    max_input_len=4096,
    max_kv_cache_capacity=8192,
)
result = llm.chat(
    [{"role": "user", "content": "Explain paged KV caches."}],
    SamplingParams(max_tokens=128, temperature=0),
)
print(result.text)
```

Pass `enable_in_flight_batching=True` together with `max_batch_size` to let
concurrent `chat()`, `generate()`, and `generate_stream()` calls from several
threads share the running batch instead of taking turns; see
[In-Flight Batching](../features/in-flight-batching.md).

Streaming uses the same runtime:

```python
for delta in llm.generate_stream(
    [{"role": "user", "content": "Write a CUDA optimization checklist."}],
    SamplingParams(max_tokens=128),
):
    print(delta.text, end="", flush=True)
```

## Start the Server

Use `trtedge serve` with a Hugging Face model ID or a local checkpoint directory:

```bash
trtedge serve Qwen/Qwen3.5-0.8B \
  --cache-dir /data/edgellm-cache \
  --max-input-len 4096 \
  --max-kv-cache-capacity 8192 \
  --port 8000
```

Add `--enable-in-flight-batching --max-batch-size 4` to serve up to four
requests at once; see [Runtime Concurrency](#runtime-concurrency).

`--max-kv-cache-capacity` is the limit for one sequence, prompt plus generated
tokens. The engine allocates a KV cache pool of `--max-batch-size` times this
capacity when it loads, so KV memory grows with both values.

See the [`trtedge` command line interface](../getting_started/cli.md) for
command help and other model tools.

The cache contains downloaded checkpoints and complete, profile-specific
runtime bundles. A launch reuses a bundle only when the base checkpoint,
optional draft checkpoint, and build profile all match. A cache miss runs
`trtedge build --components all` internally and publishes the
completed bundle atomically. The builder's default policy externalizes the
weight kinds supported by each component. Direct engine and ONNX paths are
rejected so the server cannot lose the checkpoint-to-runtime association.

Compiled bundles use a 50 GiB least-recently-used cache by default. Set
`--engine-cache-max-size-gb` to another positive limit. Use
`--clear-engine-cache` to remove compiled bundles before launch while retaining
downloaded checkpoints. Moving or copying an unchanged checkpoint does not
invalidate its cached bundle. Python callers can perform cache maintenance with
`tensorrt_edgellm.server.clear_engine_cache()` and `prune_engine_cache()`.

## Per-Layer Embedding Storage

PLE-enabled models default to full GPU residency. To use a bounded pool, pass
`--ple-residency cached --ple-cache-slots 4097`. The slot count includes the zero
slot and must be at least two. `--ple-file /data/model-shard.safetensors` selects
an original checkpoint file containing the recorded PLE tensor; otherwise the
runtime uses the model checkpoint. Cached sources must remain accessible and
unchanged while the runtime is alive.

For each engine execution, the union of distinct valid PLE rows across all
requests in that batch must fit in `cache_slots - 1`; slot 0 is reserved for
padding. For example, 4097 slots can hold at most 4096 distinct valid rows for
one execution batch. All those rows stay resident until execution completes.
If the batch does not fit, increase the pool capacity or reduce the execution
chunk/batch size. Smaller I/O transfer batches do not relax this requirement.
For a PLE that gathers several rows per token (for example Qwen3.8-Flash-Next),
run `sync; echo 3 > /proc/sys/vm/drop_caches` before the first inference;
otherwise loading the weights can take several minutes longer.

Python callers can pass `ple_config={"residency": "cached", "cache_slots": 4097,
"file_path": "/data/model-shard.safetensors"}` to `LLM`, or use
`tensorrt_edgellm.server.PleRuntimeConfig` with the same fields. These settings also
apply with `enable_in_flight_batching=True`; they do not change the engine build
or its cache key. Native Python `LLMRuntime` and `RequestEngine` accept the native
`PleRuntimeConfig` through their `ple_config` argument. Cached PLE currently
requires a PLE-enabled, single-device deployment; diffusion deployments use
resident PLE.

PLE host I/O buffers share a fixed 64 MiB budget across the runtime's local
ranks. Initialization reduces transfer batches or reader buffers to fit and
warns when this may increase loading latency; GPU cache capacity is unchanged.
Resident preload returns its budget after completion. If even the minimum
working buffers cannot fit the remaining budget, initialization fails. This
limit excludes cache/request metadata, thread stacks, and the OS page cache;
it is not a limit on total host memory usage.

## Speculative Decoding

`--speculative-config` accepts `method`, `model`, `num_speculative_tokens`,
`draft_top_k`, and `verify_tokens`. The base, draft, and auxiliary model components are
built and cached as one paired runtime. When omitted, the server reads the
proposal length from the draft checkpoint. For EAGLE3 and MTP,
`num_speculative_tokens` is the number of sequential draft steps. For DFlash,
JetSpec, and Domino it is the number of proposal tokens excluding the anchor, and must
not exceed `block_size - 1`. For dSpark it is the number of proposal tokens,
with one additional base-verification token. `draft_top_k` and `verify_tokens`
set the tree fanout and verify-node count; they take precedence over
`--draft-top-k` and `--spec-verify-tokens`. For DFlash, `verify_tokens` also
sizes the base engine when `--max-spec-verify-tokens` is not set.

EAGLE3 example:

```bash
trtedge serve Qwen/Qwen3-1.7B \
  --cache-dir /data/edgellm-cache \
  --speculative-config \
  '{"method":"eagle3","model":"AngelSlim/Qwen3-1.7B_eagle3","num_speculative_tokens":3}'
```

DFlash example:

```bash
trtedge serve Qwen/Qwen3.5-4B \
  --cache-dir /data/edgellm-cache \
  --speculative-config \
  '{"method":"dflash","model":"z-lab/Qwen3.5-4B-DFlash","num_speculative_tokens":3}'
```

Muse-Glimmer DFlash and DFlash2 examples:

```bash
trtedge serve meta-models/Muse-Glimmer-30B \
  --cache-dir /data/edgellm-cache \
  --speculative-config \
  '{"method":"dflash","model":"meta-models/Muse-Glimmer-30B-assistant"}'

trtedge serve meta-models/Muse-Glimmer-30B \
  --cache-dir /data/edgellm-cache \
  --speculative-config \
  '{"method":"dflash","model":"incoai/Muse-Glimmer-30B-DFlash2"}'

trtedge serve RadixArk/Muse-Glimmer-NVFP4 \
  --cache-dir /data/edgellm-cache \
  --speculative-config \
  '{"method":"dflash","model":"incoai/Muse-Glimmer-30B-DFlash2"}'
```

DFlash2 DDTree example (greedy requests verify a 32-node tree with four children
per node; sampled requests run on the vanilla decoder):

```bash
trtedge serve Qwen/Qwen3.8-27B \
  --cache-dir /data/edgellm-cache \
  --speculative-config \
  '{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","draft_top_k":4,"verify_tokens":32}'
```

JetSpec example:

```bash
trtedge serve Qwen/Qwen3-8B \
  --cache-dir /data/edgellm-cache \
  --speculative-config \
  '{"method":"jetspec","model":"JetSpec/jetspec-qwen3-8b","num_speculative_tokens":15}'
```

Domino example:

```bash
trtedge serve Qwen/Qwen3-8B \
  --cache-dir /data/edgellm-cache \
  --speculative-config \
  '{"method":"domino","model":"Huang2020/Qwen3-8B-Domino-b16"}'
```

dSpark example:

```bash
trtedge serve Qwen/Qwen3-4B \
  --cache-dir /data/edgellm-cache \
  --speculative-config \
  '{"method":"dspark","model":"deepseek-ai/dspark_qwen3_4b_block7","num_speculative_tokens":7}'
```

For a checkpoint containing native MTP layers, select `mtp` without `model`.
Gemma MTP instead supplies its separate assistant checkpoint as `model`. The
server defaults MTP, DFlash, JetSpec, Domino, and dSpark to their linear contracts.
Where the method supports branching, setting `--draft-top-k` above 1 selects
its tree contract at runtime. The fixed-ABI base artifact supports both linear
and tree verification within its built capacities. The
`--spec-draft-step`, `--spec-proposal-tokens`, and `--spec-verify-tokens`
options override the corresponding runtime geometry when needed. Use
`--max-spec-draft-tokens` and `--max-spec-verify-tokens` to choose the draft
and target build capacities. The
`disable_spec_decode` request field can disable drafting for one request,
except with a Gemma MTP verification engine; use a standalone target bundle
for target-only Gemma inference.
See [Logit Bias](../format/input-format.md#logit-bias) for speculative-decoding
behavior and validation limits.

`--max-spec-verify-tokens` defaults to 60 for MTP and EAGLE, and memory grows
with it. To save memory, set it to what drafting needs, especially with MTP
chain drafting: the number of draft tokens plus one, for example 4 for
`num_speculative_tokens: 3`. Jump-forward is then limited to that width.

## Streaming Metrics

`LLM.generate_stream()` returns the public `tensorrt_edgellm.server.CancellableIterator`.
Call `cancel()` to request cancellation without closing an executing generator.
Call `close()` to cancel and close it; if another thread is iterating, wait for
that thread before releasing its resources.

Adapters can wrap transformed deltas with
`CancellableIterator(transformed_generator, source_stream.cancel)`. Close the
source stream in the transformed generator's `finally` block so early closure
also drains the native producer.

The [interactive chat frontend](interactive-chat.md#timings), `trtedge chat`,
uses this API to display per-response throughput, context usage, and latency.
It calls `LLM.generate_stream()` directly with an `on_metrics` callback. The
HTTP server's `EngineClient` does not supply this callback or collect these
metrics.

Pass a function as `on_metrics` to `llm.generate_stream()`. The API calls that
function once with the final metrics after the producer stops and the stream
is drained or closed, including after cancellation. The loop receives
`StreamDelta` objects; the callback receives a separate `GenerationMetrics`
object. Collection is disabled when `on_metrics` is omitted.

These per-request timings measure elapsed wall-clock time. Each interval
includes CPU work, GPU execution, and waiting within that interval. They are
independent of `set_profiling_enabled()`, which controls aggregate counters and
GPU stage timing measured with CUDA events. See the
[chat timing definitions](interactive-chat.md#timings) for the measurement
boundaries.

Native callers opt in with `StreamChannel::create(true)` (Python:
`StreamChannel.create(collect_metrics=True)`).

```python
def report_metrics(metrics):
    print(f"\nOutput tokens: {metrics.generated_tokens}")
    print(f"Decode tokens/s: {metrics.decode_tokens_per_second:.2f}")

for delta in llm.generate_stream(
    [{"role": "user", "content": "Hello!"}], on_metrics=report_metrics
):
    print(delta.text, end="", flush=True)
```

The loop prints the response text. When generation ends, the API calls
`report_metrics(metrics)` to print the final token count and throughput.

`generated_tokens` counts all output tokens, including any sampled during
prefill. `decoded_tokens` counts only tokens produced after prefill, matching
the interval measured by `generation_time`.
Use `decode_tokens_per_second` for decode throughput; dividing the total output
count by `generation_time` overstates throughput when prefill produces a token.
An output completed during prefill has zero decode tokens, time, and throughput.
The throughput property also returns zero when decode timing is unavailable.

The existing `LLMGenerationMetrics.generated_tokens` profiler counter sums
output tokens across sequences and recorded runs. `StreamMetrics.generated_tokens`
counts output tokens for one sequence. Both include prefill output; their
scope differs. The callback exposes the per-sequence count as
`GenerationMetrics.generated_tokens`.

## KV Cache Reuse

Context reuse is disabled by default. Enable it when constructing the server:

```bash
trtedge serve Qwen/Qwen3.5-0.8B \
  --cache-dir /data/edgellm-cache \
  --enable-context-reuse \
  --context-cache-max-records 1024
```

The Python API accepts the same deployment configuration and exposes native
reuse metrics:

```python
from tensorrt_edgellm.server import ContextCacheConfig, LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3.5-0.8B",
    context_cache_config=ContextCacheConfig(enabled=True, max_records=1024),
)
llm.generate("shared prefix", SamplingParams(max_tokens=32))
llm.generate("shared prefix with another suffix", SamplingParams(max_tokens=32))
print(llm.get_context_cache_metrics().reused_tokens)
```

Two `SamplingParams` fields control reuse for one request:

- `reuse_context=False` skips lookup and publication.
- `cache_generated_tokens=False` publishes only the prefill endpoint. When
  omitted, the request also publishes its generated tokens.

A speculative request that reuses context on a hybrid model such as Qwen3.5 can
publish only the prefill endpoint. Its `cache_generated_tokens` defaults to
`False`, and setting it to true is an error (HTTP 400 on the server).

The OpenAI chat and completion requests take the same `reuse_context` and
`cache_generated_tokens` fields.

See the authoritative [KV Cache Reuse support matrix](../features/kv-cache-reuse.md#support-matrix)
for supported model and speculative-decoding contracts.

A hybrid model such as Qwen3.5 also keeps snapshots of its recurrent state,
by default four times `--max-batch-size` in total. See [Snapshots](../features/kv-cache-reuse.md#snapshots)
to keep more or fewer.

With a chat template that folds a thinking marker into the generation prompt
(Qwen3 under `enable_thinking=false`, for example), the server publishes the
checkpoint before that marker and replays the few unstable tail tokens, so the
record stays valid as a prefix of the next turn.

For encoder-cache behavior and server configuration limits, see
[Encoder Embedding Cache](../features/kv-cache-reuse.md#configuring-the-encoder-embedding-cache).

For tenant isolation, follow the [cache-domain requirements](../features/kv-cache-reuse.md#cache-domain).

## OpenAI Chat

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages":[{"role":"user","content":"Hello"}],
    "max_tokens":128
  }'
```

Set `stream=true` for SSE chunks ending in `data: [DONE]`. Set
`stream_options.include_usage=true` for a final usage chunk. Usage is taken
from the native generation result; normal inference does not run a second
tokenization pass.

The request contract includes sampling, stop strings, log probabilities,
`logit_bias`, tools, `parallel_tool_calls`, thinking, structured output, and
per-request speculative disablement. `ignore_eos=true` keeps generating
past the end-of-sequence token until `max_tokens`, for fixed-length load tests.
`seed` accepts an unsigned 64-bit request
seed: a seeded request draws its own random stream and replays exactly whatever
else is in the batch. A request without `seed` draws a fixed stream keyed on its
batch row and decode step, so it repeats across runs when served alone, while
requests sharing a batch draw different values. Set `seed` for evaluations.
`top_k=-1` disables top-k filtering. `min_p` and text
`repetition_penalty` are accepted for client compatibility, but the runtime
currently supports only their defaults (`0.0` and `1.0`); non-default values
are normalized to those defaults with a server warning. Nonzero
`frequency_penalty` and `presence_penalty` remain unsupported and are rejected.
Only `n=1` is supported.
Provider chat templating and the assistant generation prompt are enabled by
default. Set `apply_chat_template=false` only for an already formatted prompt,
or `add_generation_prompt=false` when continuing an existing assistant turn.

### Structured Output

`response_format` constrains the reply to a JSON object or a JSON Schema:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages":[{"role":"user","content":"Give me a person record."}],
    "response_format":{
      "type":"json_schema",
      "json_schema":{"name":"person","schema":{
        "type":"object",
        "properties":{"name":{"type":"string"},"age":{"type":"integer"}},
        "required":["name","age"]}}}
  }'
```

`response_format` covers `text`, `json_object`, and `json_schema` — the whole of
the OpenAI specification. For `regex`, `ebnf`, `structural_tag`, and `choice`,
use the low-level `guided_decoding` field, which takes the same shape as in a
request file:

```bash
  -d '{
    "messages":[{"role":"user","content":"Answer yes or no."}],
    "guided_decoding":{"choice":["yes","no"]}
  }'
```

Setting both surfaces on one request is rejected rather than one silently
winning. Streaming works normally, and the assembled output satisfies the guide.
A schema keyword that the backend accepts but cannot enforce is rejected with a
400 naming the keyword. See [Guided Decoding](../features/guided-decoding.md).

Add `"jump_forward": true` inside `guided_decoding` to append grammar-forced
tokens without forward passes. It is off by default and has no `response_format`
equivalent. See [Jump-forward](../features/guided-decoding.md#jump-forward).

### Tool Calls and Thinking

Launch automatic tool choice explicitly:

```bash
trtedge serve Qwen/Qwen3.5-0.8B \
  --cache-dir /data/edgellm-cache \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
```

Requests use OpenAI `tools`, `tool_choice`, assistant `tool_calls`, and matching
`tool` messages. Parsed thinking is returned as `reasoning_content` only when
the request sets `enable_thinking=true` or
`chat_template_kwargs.enable_thinking=true`.
Streaming responses emit indexed tool-call deltas as soon as each generated
call is complete and end with `finish_reason="tool_calls"`.

Qwen3.8's provider template accepts `reasoning_effort` values `xhigh` (the
provider default), `medium`, and `low` when thinking is enabled. Edge-LLM
passes the value through unchanged, so an unsupported value fails template
rendering instead of being remapped. Because Edge-LLM defaults to non-thinking
requests, set `enable_thinking=true` together with the desired effort.

### Image, Video, and Audio Input

Serving a multimodal checkpoint builds and attaches its model-specific visual
and audio components automatically:

```bash
trtedge serve Qwen/Qwen3-VL-2B-Instruct \
  --cache-dir /data/edgellm-cache \
  --allowed-local-media-path /data/media
```

Muse-Glimmer image and video chat use its FP16 multimodal checkpoint; the
NVFP4 checkpoint is text-only:

```bash
trtedge serve meta-models/Muse-Glimmer-30B \
  --cache-dir /data/edgellm-cache \
  --max-image-tokens 4096 \
  --max-image-tokens-per-image 4096 \
  --allowed-local-media-path /data/media
```

Use an `image_url` block for an image or a `video_url` block for a video.
Muse-Glimmer samples video at 2 FPS, up to 96 frames; `fps` and `nframes` on
the video content block override those defaults.

OpenAI content blocks accept `image_url`, `video_url`, `input_audio`, and
`audio_url` forms described in [Input Format](../format/input-format.md). Data
URLs and files under `--allowed-local-media-path` are supported. Remote HTTP
and HTTPS sources are downloaded with per-modality size limits and a bounded
timeout. Local paths remain disabled unless they are under the configured
allowed path.

Nemotron Omni video uses its checkpoint's video patch embedder and dynamic
aspect-preserving frame grids:

```bash
trtedge serve \
  nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
  --cache-dir /data/edgellm-cache \
  --allowed-local-media-path /data/media
```

Each Nemotron Omni request accepts one video and no additional images. Use
either `fps` or `nframes`; the server samples the clip, validates the visual
engine profile before decoding, and rejects frame lists that exceed its raw
pre-pruning tubelet capacity.

ASR-capable autoregressive models expose transcription:

```bash
curl -s http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=qwen3-asr \
  -F response_format=json
```

Uploads are limited to 25 MiB. The response format can be `json` or `text`.
Nemotron-3.5-ASR is a non-autoregressive RNN-T model with a separate
server; see [Nemotron-3.5-ASR](../../developer_guide/models/nemotron3_5_asr.md).

### Omni Audio Output

When a model owns Talker, CodePredictor, and Code2Wav components, chat can
return text and PCM audio:

```json
{
  "messages": [{"role": "user", "content": "Say hello"}],
  "modalities": ["text", "audio"],
  "audio": {"voice": "Ryan", "format": "pcm16"}
}
```

The same runtime exposes `POST /v1/audio/speech` and `GET /v1/voices`. Speech
output is 24 kHz, mono, signed 16-bit little-endian PCM. Both chat audio and
speech requests accept `talker_temperature`, `talker_top_k`, `talker_top_p`,
`repetition_penalty`, `max_audio_length`, `codec_chunk_frames`, and
`talker_prefill_threshold` alongside `voice`.

A TTS-only checkpoint uses the same model/cache contract from Python:

```python
from tensorrt_edgellm.server import TTS

tts = TTS(
    model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    cache_dir="/data/edgellm-cache",
)
tts.serve(port=8000)
```

## OpenAI Text Completions

`/v1/completions` accepts one raw string or one list of token IDs. Token IDs
bypass chat formatting and tokenization. Include all required special tokens in
the list because the server does not add BOS, EOS, or chat-template tokens.

```bash
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":[9707,11],"max_tokens":32,"stream":true}'
```

Streaming uses SSE and supports a final usage chunk through
`stream_options.include_usage=true`.

## Anthropic Messages

`POST /v1/messages` and `POST /v1/messages/count_tokens` share the loaded model
and admission controller with OpenAI chat. The adapter supports system prompts,
text, base64 image blocks, client tools, `tool_use`/`tool_result`, thinking
output, and Anthropic SSE framing. Server-executed tools are not run by
TensorRT Edge-LLM. Launch Claude Code from
[Use Claude Code](integration/claude-code.md).

When `--api-key` is set, both OpenAI bearer authentication and Anthropic
`x-api-key` authentication are accepted.

## Endpoints

| Method | Path | Contract |
|---|---|---|
| `GET` | `/health`, `/health/ready` | Runtime, queue, and capability state; under in-flight batching `/health` also reports the scheduler counters (`scheduling`) and the engine state (`engine`), and both return 503 once the engine no longer serves |
| `GET` | `/v1/models` | The loaded model |
| `POST` | `/v1/chat/completions` | OpenAI chat and SSE |
| `POST` | `/v1/responses` | OpenAI Responses, translated to chat |
| `POST` | `/v1/completions` | OpenAI text completion and SSE |
| `POST` | `/v1/messages` | Anthropic Messages and SSE |
| `POST` | `/v1/messages/count_tokens` | Explicit input token count |
| `POST` | `/v1/audio/transcriptions` | ASR upload |
| `POST` | `/v1/audio/speech` | Omni/TTS PCM stream |
| `GET` | `/v1/voices` | Available speakers |

`POST /v1/responses` translates a Responses request onto chat completions so
[Codex](integration/codex.md) can use the server. `stream: true` emits each
text and tool-call delta as the model generates it. The server does not expose
embeddings, tokenization, or a metrics HTTP endpoint. Context-cache metrics
are available from the Python API.

## Runtime Concurrency

By default the server admits one request at a time: the runtime has one mutable
generation state, and a bounded async queue configured by
`--max-queued-requests` and `--queue-timeout` holds the rest. Queue overflow and
timeout return HTTP 429 (Anthropic 529). Streaming disconnects cancel the native
channel immediately, wait for the native worker to exit, and then release the
runtime lease. Engines stay resident across HTTP connections; graceful server
shutdown drains active work and releases the runtime and its device resources.

`--enable-in-flight-batching` replaces the one-at-a-time path with the request
engine: up to `--max-batch-size` requests decode together, and a new request
joins the running batch at the next generation boundary instead of waiting for
it to finish. Admission is then a limit rather than a queue: at most
`--max-batch-size` plus `--max-queued-requests` requests are in flight, and a
request past that limit gets an immediate 429. A disconnect cancels the request
inside the engine, so an abandoned stream stops decoding and frees its seat.
`/health` reports `in_flight_batching: true`, `max_num_seqs` equal to the batch
size, and a `scheduling` block with the engine's counters (submitted, completed,
cancelled, admissions that joined mid-flight, and the reasons a queued request
had to wait), and an `engine` block with the scheduling policy in effect. Once
the engine fails every request, `/health` and `/health/ready` return HTTP 503
with the reason, and so does every new request (error type
`engine_unavailable`); see
[Reading `/health`](../features/in-flight-batching.md#reading-health) and
[Errors](../features/in-flight-batching.md#errors).
`--min-request-priority` (default 0, at most 0) refuses a request whose
`priority` is lower with HTTP 400, with or without the flag.

Not every deployment can take the flag, and not every request can join a
running batch; see [In-Flight Batching](../features/in-flight-batching.md) for
the support matrix and the
[scheduling modes](../features/in-flight-batching.md#scheduling-modes)
(`--ifb-scheduling-mode`, chunked prefill, and priorities). Tensor parallelism
remains on the one-at-a-time path.
