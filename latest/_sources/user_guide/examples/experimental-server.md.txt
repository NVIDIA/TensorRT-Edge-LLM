# Experimental Python API and Server

The experimental Python frontend accepts a supported checkpoint, builds all of
its runtime components directly, and exposes offline generation plus OpenAI and
Anthropic HTTP APIs. ONNX is not used by this path.

> **Status:** Experimental. Only the API contracts and launch options described
> here are supported.

## Install

Build TensorRT Edge-LLM with Python bindings, then install the package and
server dependencies in the same environment:

```bash
cd /path/to/TensorRT-Edge-LLM
pip install -e ".[server,server-tools,native-build]"
```

For a prebuilt TensorRT Edge-LLM package that already contains the Python
runtime extension, omit `native-build`. `server-tools` is required only for
model-native tool chat templates; plain text and multimodal serving use the
smaller `server` extra. None of these extras installs the PyTorch/ONNX exporter.

## Python API

`model` is a Hugging Face model ID or a local checkpoint directory. On a cache
miss, one builder invocation compiles every component owned by the model
family. Supported weights stay outside the engines and are loaded once from
the resolved checkpoint during runtime initialization.

```python
from experimental.server import LLM, SamplingParams

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

Streaming uses the same runtime:

```python
for delta in llm.generate_stream(
    [{"role": "user", "content": "Write a CUDA optimization checklist."}],
    SamplingParams(max_tokens=128),
):
    print(delta.text, end="", flush=True)
```

## Start the Server

Pass the model checkpoint, not an engine path:

```bash
tensorrt-edgellm-serve Qwen/Qwen3.5-0.8B \
  --cache-dir /data/edgellm-cache \
  --max-input-len 4096 \
  --max-kv-cache-capacity 8192 \
  --port 8000
```

The cache contains downloaded checkpoints and complete, profile-specific
runtime bundles. A launch reuses a bundle only when the base checkpoint,
optional draft checkpoint, and build profile all match. A cache miss runs
`tensorrt-edgellm-build --components all --externalize-weights all` internally
and publishes the completed bundle atomically. Direct engine and ONNX paths are
rejected so the server cannot lose the checkpoint-to-runtime association.

Compiled bundles use a 50 GiB least-recently-used cache by default. Set
`--engine-cache-max-size-gb` to another positive limit. Use
`--clear-engine-cache` to remove compiled bundles before launch while retaining
downloaded checkpoints. Moving or copying an unchanged checkpoint does not
invalidate its cached bundle. Python callers can perform cache maintenance with
`experimental.server.clear_engine_cache()` and `prune_engine_cache()`.

## Speculative Decoding

`--speculative-config` accepts `method`, `model`, and
`num_speculative_tokens`. The base, draft, and auxiliary model components are
built and cached as one paired runtime. When omitted, the server reads the
proposal length from the draft checkpoint. For EAGLE3 and MTP,
`num_speculative_tokens` is the number of sequential draft steps. For DFlash
and JetSpec it is the proposal block size; for dSpark it is the number of
proposal tokens, with one additional base-verification token.

EAGLE3 example:

```bash
tensorrt-edgellm-serve Qwen/Qwen3-1.7B \
  --cache-dir /data/edgellm-cache \
  --speculative-config \
  '{"method":"eagle3","model":"AngelSlim/Qwen3-1.7B_eagle3","num_speculative_tokens":3}'
```

DFlash example:

```bash
tensorrt-edgellm-serve Qwen/Qwen3.5-4B \
  --cache-dir /data/edgellm-cache \
  --speculative-config \
  '{"method":"dflash","model":"z-lab/Qwen3.5-4B-DFlash","num_speculative_tokens":3}'
```

JetSpec example:

```bash
tensorrt-edgellm-serve Qwen/Qwen3-8B \
  --cache-dir /data/edgellm-cache \
  --speculative-config \
  '{"method":"jetspec","model":"JetSpec/jetspec-qwen3-8b","num_speculative_tokens":16}'
```

dSpark example:

```bash
tensorrt-edgellm-serve Qwen/Qwen3-4B \
  --cache-dir /data/edgellm-cache \
  --speculative-config \
  '{"method":"dspark","model":"deepseek-ai/dspark_qwen3_4b_block7","num_speculative_tokens":7}'
```

For a checkpoint containing native MTP layers, select `mtp` without `model`.
Gemma MTP instead supplies its separate assistant checkpoint as `model`. The
server defaults MTP, DFlash, JetSpec, and dSpark to their linear contracts.
Where the method supports branching, setting `--draft-top-k` above 1 selects
its tree contract and causes the direct builder to compile matching tree-base
inputs automatically. The
`disable_spec_decode` request field can disable drafting for one request,
except with a Gemma MTP verification engine; use a standalone target bundle
for target-only Gemma inference.
See [Logit Bias](../format/input-format.md#logit-bias) for speculative-decoding
behavior and validation limits.

## KV Cache Reuse

Context reuse is disabled by default. Enable it when constructing the server:

```bash
tensorrt-edgellm-serve Qwen/Qwen3.5-0.8B \
  --cache-dir /data/edgellm-cache \
  --enable-context-reuse \
  --context-cache-max-records 1024
```

The Python API accepts the same deployment configuration and exposes native
reuse metrics:

```python
from experimental.server import ContextCacheConfig, LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3.5-0.8B",
    context_cache_config=ContextCacheConfig(enabled=True, max_records=1024),
)
llm.generate("shared prefix", SamplingParams(max_tokens=32))
llm.generate("shared prefix with another suffix", SamplingParams(max_tokens=32))
print(llm.get_context_cache_metrics().reused_tokens)
```

`SamplingParams(reuse_context=False)` bypasses lookup and publication for a
request. `cache_generated_tokens=False` publishes only the prefill endpoint.
The OpenAI chat request exposes the same controls as strict boolean
`reuse_context` and `cache_generated_tokens` fields.

See the authoritative [KV Cache Reuse support matrix](../features/kv-cache-reuse.md#support-matrix)
for supported model and speculative-decoding contracts. Configure the server's
recurrent snapshot pools with
`--context-cache-recurrent-snapshot-pool-bytes` and
`--context-cache-partial-kv-snapshot-pool-bytes`.

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
`logit_bias`, tools, `parallel_tool_calls`, thinking, and per-request
speculative disablement. Unsupported fields such as penalties, seed, and
structured output are rejected rather than ignored. Only `n=1` is supported.

### Tool Calls and Thinking

Launch automatic tool choice explicitly:

```bash
tensorrt-edgellm-serve Qwen/Qwen3.5-0.8B \
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

### Image, Video, and Audio Input

Serving a multimodal checkpoint builds and attaches its model-specific visual
and audio components automatically:

```bash
tensorrt-edgellm-serve Qwen/Qwen3-VL-2B-Instruct \
  --cache-dir /data/edgellm-cache \
  --allowed-local-media-path /data/media
```

OpenAI content blocks accept `image_url`, `video_url`, `input_audio`, and
`audio_url` forms described in [Input Format](../format/input-format.md). Data
URLs and files under `--allowed-local-media-path` are supported. Remote HTTP
and HTTPS sources are downloaded with per-modality size limits and a bounded
timeout. Local paths remain disabled unless they are under the configured
allowed path.

Nemotron Omni video uses its checkpoint's video patch embedder and dynamic
aspect-preserving frame grids:

```bash
tensorrt-edgellm-serve \
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
experimental server; see [Nemotron-3.5-ASR](../../developer_guide/models/nemotron3_5_asr.md).

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
from experimental.server import TTS

tts = TTS(
    model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    cache_dir="/data/edgellm-cache",
)
tts.serve(port=8000)
```

## Anthropic and Claude Code

`POST /v1/messages` and `POST /v1/messages/count_tokens` share the loaded model
and admission controller with OpenAI chat. The adapter supports system prompts,
text, base64 image blocks, client tools, `tool_use`/`tool_result`, thinking
output, and Anthropic SSE framing. Server-executed tools are not run by
TensorRT Edge-LLM.

Launch with a stable served name, then point Claude Code at the server:

```bash
tensorrt-edgellm-serve Qwen/Qwen3.5-0.8B \
  --cache-dir /data/edgellm-cache \
  --served-model-name qwen35-local

ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_API_KEY=local \
ANTHROPIC_DEFAULT_OPUS_MODEL=qwen35-local \
ANTHROPIC_DEFAULT_SONNET_MODEL=qwen35-local \
ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen35-local \
claude
```

When `--api-key` is set, both OpenAI bearer authentication and Anthropic
`x-api-key` authentication are accepted.

## Endpoints

| Method | Path | Contract |
|---|---|---|
| `GET` | `/health`, `/health/ready` | Runtime, queue, and capability state |
| `GET` | `/v1/models` | The loaded model |
| `POST` | `/v1/chat/completions` | OpenAI chat and SSE |
| `POST` | `/v1/messages` | Anthropic Messages and SSE |
| `POST` | `/v1/messages/count_tokens` | Explicit input token count |
| `POST` | `/v1/audio/transcriptions` | ASR upload |
| `POST` | `/v1/audio/speech` | Omni/TTS PCM stream |
| `GET` | `/v1/voices` | Available speakers |

The server does not expose Completions, Responses, embeddings, tokenization, or
a metrics HTTP endpoint. Context-cache metrics are available from the Python
API.

## Runtime Concurrency

The current high-level runtime has one mutable generation state. The server
therefore admits one request at a time and uses a bounded async queue configured
by `--max-queued-requests` and `--queue-timeout`. Queue overflow and timeout
return HTTP 429 (Anthropic 529). Streaming disconnects cancel the native channel
immediately, wait for the native worker to exit, and then release the runtime
lease. Engines stay resident across HTTP connections; graceful server shutdown
drains active work and releases the runtime and its device resources.

Continuous batching, chunked prefill scheduling, and tensor parallelism require
additional native scheduler support and are rejected at launch.
