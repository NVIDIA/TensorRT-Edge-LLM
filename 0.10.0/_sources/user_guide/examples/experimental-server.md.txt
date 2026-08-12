# Experimental High-Level Python API and Server

The experimental Python API wraps export, engine build, engine loading, generation, streaming, and OpenAI-compatible serving.

> **Status:** Experimental. API may change between releases.

## Prerequisites

Complete the [Installation Guide](../getting_started/installation.md) with the C++ runtime, Python bindings, and server dependencies enabled before proceeding. The examples below assume `experimental.server` and `tensorrt_edgellm` are importable from the active Python environment.

If the active environment was installed with base export dependencies only, install the server dependencies before building Python bindings or launching the server:

```bash
cd /path/to/TensorRT-Edge-LLM
python -m pip install -e ".[server]"
```

## Python API

From a HuggingFace checkpoint:

```python
from experimental.server import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-1.7B")
outputs = llm.generate(
    ["What is the capital of France?"],
    SamplingParams(temperature=0.7, max_tokens=128),
)
print(outputs[0].text)
```

From existing ONNX or engine directories:

```python
from experimental.server import LLM

llm = LLM(onnx_dir="/path/to/llm_onnx")
llm = LLM(engine_dir="/path/to/llm_engine")
```

Streaming:

```python
from experimental.server import LLM, SamplingParams

llm = LLM(engine_dir="/path/to/llm_engine")

for delta in llm.generate_stream(
    [{"role": "user", "content": "Tell me a story."}],
    SamplingParams(max_tokens=256),
):
    print(delta.text, end="", flush=True)
```

## OpenAI-Compatible Server

```bash
python -m experimental.server \
  --model Qwen/Qwen3-1.7B \
  --port 8000
```

Serve an existing engine without exporting or building:

```bash
python -m experimental.server \
  --model /path/to/llm_engine \
  --port 8000
```

For a multimodal model, point the server at the encoders explicitly:

- `--multimodal-engine-dir` (alias `--visual-engine-dir`): prebuilt visual
  and/or audio encoder engines for a prebuilt `--model` engine dir.
- `--visual-onnx-dir` / `--audio-onnx-dir`: prebuilt encoder ONNX dirs when
  `--model` is a prebuilt ONNX dir.

Query:

```bash
curl -sN http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 128}'
```

Streaming query:

```bash
curl -sN http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 128, "stream": true}'
```

Legacy raw-prompt completion (no chat template applied):

```bash
curl -sN http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 128}'
```

Tool-aware query:

```bash
curl -sN http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto",
    "max_tokens": 128
  }'
```

To continue an agentic loop, include the previous assistant `tool_calls` and
the matching `tool` response messages in the next request.

Tool response follow-up:

```json
{
  "messages": [
    {"role": "user", "content": "What is the weather in Paris?"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\":\"Paris\"}"
        }
      }]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "content": "{\"temperature\":22,\"unit\":\"celsius\"}"
    }
  ],
  "tools": [{
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object"}}
  }]
}
```

## Common Inputs

`LLM` requires exactly one source:

| Source | Meaning |
|---|---|
| `model` | HuggingFace model ID or local checkpoint; export, build, then load |
| `onnx_dir` | Existing ONNX directory; build then load |
| `engine_dir` | Existing engine directory; load only |

Encoders are passed alongside: `visual_onnx_dir` / `audio_onnx_dir` with
`onnx_dir`, and `multimodal_engine_dir` (alias `visual_engine_dir`) with
`engine_dir`. Models that support audio: Qwen3-Omni, Qwen3-ASR, Nemotron-Omni.

## Audio Input

The server accepts three OpenAI-compatible content forms inside user
messages for models that support audio (Qwen3-Omni, Qwen3-ASR, Nemotron-Omni):

```json
{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}
{"type": "audio_url", "audio_url": {"url": "file:///abs/path | data:audio/...;base64,..."}}
{"type": "audio", "audio": "<local path>"}
```

`http(s)://` URLs are rejected by design — inline the bytes as base64 via
`input_audio`, or pass a local path with the server started as
`--allowed-local-media-path <dir>` (local paths and `file://` are refused over
HTTP otherwise, and are confined to that directory when it is set). Supported
containers: `.wav`, `.mp3`, `.flac`. The server decodes the container
in-process via vendored miniaudio and the audio runner extracts the
mel-spectrogram in C++ (no HF `transformers` feature extractor or Python
preprocessing step is required). The model-appropriate feature extractor
is selected automatically from the engine's `audio/config.json::model_type`:
`whisper` for Qwen3-Omni / Qwen3-ASR, `parakeet` for Nemotron-Omni.

Example (base64-inline):

```bash
B64=$(base64 -w0 sample.wav)
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":[ \
       {\"type\":\"input_audio\",\"input_audio\":{\"data\":\"$B64\",\"format\":\"wav\"}}, \
       {\"type\":\"text\",\"text\":\"Transcribe.\"}]}],\"max_tokens\":128}"
```

## Audio Output (Qwen3-Omni)

For Qwen3-Omni models the server can stream synthesized speech alongside the
text response, following the OpenAI chat-completions audio schema. It requires
the Omni audio-output engines (Talker, CodePredictor, Code2Wav) placed as
siblings of the Thinker engine directory — they are auto-detected at startup:

```
{engine_root}/
    thinker/          # engine_dir passed to LLM (llm.engine)
    talker/           # llm.engine
    code_predictor/   # llm.engine
    code2wav/         # code2wav.engine
```

The dirs can also be passed explicitly via `LLM(talker_engine_dir=...,
code_predictor_engine_dir=..., code2wav_engine_dir=...)`.

> **Build requirement:** the Python bindings must be built with the CuTe DSL
> GEMM enabled, or the Talker MLP fails and no audio is produced. Generate the
> AOT artifact first, then build with `ENABLE_CUTE_DSL=gemm`:
>
> ```bash
> python kernelSrcs/build_cutedsl.py --kernels gemm --gpu_arch <sm>
> TRT_PACKAGE_DIR=... ENABLE_CUTE_DSL=gemm \
>     python experimental/server/setup_pybind.py build_ext --inplace
> ```

Request audio by adding `modalities` (and optionally an `audio` object):

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "local", "stream": true,
       "modalities": ["text", "audio"],
       "audio": {"format": "pcm16", "voice": ""},
       "messages": [{"role": "user", "content": "Introduce yourself."}]}'
```

Streaming responses interleave text and audio deltas in one SSE stream.
Audio arrives as base64 int16 mono PCM at 24 kHz:

```json
{"choices": [{"delta": {"content": "Hello"}, "index": 0}]}
{"choices": [{"delta": {"audio": {"id": "audio-...", "data": "<base64 pcm16>",
                                  "format": "pcm16", "sample_rate": 24000}}, "index": 0}]}
```

Generation is truly streaming: the Thinker and Talker run interleaved, so the
first audio chunk is emitted after roughly `talker_prefill_threshold` text
tokens rather than after the full text completes. With `stream: false` the
server aggregates the chunks and returns one `message.audio.data` blob plus
`message.audio.transcript`.

Talker knobs live inside the `audio` object (they are namespaced there to
avoid colliding with text sampling fields):

| Field | Default | Description |
|---|---:|---|
| `voice` | `""` | Speaker name; empty selects the model default |
| `format` | `"pcm16"` | Output encoding; only `pcm16` is supported |
| `codec_chunk_frames` | `10` | Vocode every N codec frames (1 frame ≈ 80 ms of audio). Smaller values lower chunk latency at the cost of more Code2Wav invocations |
| `talker_prefill_threshold` | `4` | Thinker tokens accumulated before Talker prefill starts |
| `talker_temperature` | `0.9` | Talker sampling temperature (must be > 0; greedy Talker sampling never emits EOS) |
| `talker_top_k` | `50` | Talker top-K |
| `talker_top_p` | `1.0` | Talker top-P |
| `repetition_penalty` | `1.05` | Talker codec repetition penalty |
| `max_audio_length` | `4096` | Maximum codec frames per response |

`tools` and `logprobs` are rejected (400) in combination with audio output.

## Text-to-Speech (`/v1/audio/speech`)

Any server with the audio-output engines loaded also exposes an OpenAI-style
TTS endpoint. Unlike chat with `modalities: ["audio"]`, the input text goes
straight to the Talker — no Thinker generation pass:

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "local", "input": "今天天气真不错。", "voice": ""}' \
  --output speech.pcm
```

`response_format: "pcm"` (default) streams raw int16 mono 24 kHz PCM as
chunks are vocoded (headers carry `X-Sample-Rate` / `X-Channels` /
`X-Sample-Format`); `"wav"` aggregates and returns a complete WAV file. The
Talker knobs from the table above sit at the top level of the body
(`talker_temperature`, `codec_chunk_frames`, ...). `input` is capped at 4096
characters.

For Qwen3-TTS-style engine sets (Talker + CodePredictor + Code2Wav, no text
model), serve TTS-only — chat endpoints then return 400:

```python
from experimental.server import TTS

tts = TTS(talker_engine_dir="/engines/qwen3-tts/talker")
tts.serve(port=8000)
```

`code_predictor_engine_dir` / `code2wav_engine_dir` default to the talker
directory's siblings; `tokenizer_dir` defaults to the talker directory itself
(Qwen3-TTS exports carry the tokenizer and `text_embedding.safetensors`
there). On a Qwen3-Omni server, pass the thinker engine dir as
`tokenizer_dir` — the text embedding lives there.

### Transcription endpoint (`/v1/audio/transcriptions`)

For ASR models the OpenAI transcription route accepts a multipart audio
upload and returns `{"text": ...}`. Clip duration is bounded by the engine
audio profile (~30 s, matching vLLM); errors are staged `400` (bad request)
/ `413` (too long) / `503` (busy) / `500`.

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=local -F file=@sample.wav
```

## Video Input

Video content accepts a source URL or pre-extracted frames:

```json
{"type": "video_url", "video_url": {"url": "file:///abs/path | data:video/...;base64,..."}}
{"type": "video", "video": "<local path>"}
{"type": "video", "frames": ["<frame path>", "..."], "fps": 1.0}
```

Sampling is controlled per request with `fps`, `nframes`, `min_frames` and
`max_frames`. Local paths follow the same `--allowed-local-media-path` rule as
audio; `http(s)://` is rejected.

Nemotron-Omni video has extra constraints: a request may carry at most one
video and no images alongside it, and it always runs as a batch of one (video
requests are never micro-batched). `do_resize: false` is rejected because the
runner always resizes frames to the target patch grid. Frames are resized with
UINT8 bicubic interpolation, a close but not bit-exact match to the HF FP32
antialiased resize.

> **Build requirement:** the Nemotron-Omni patch embedder runs a CuTe DSL FP16
> GEMM in the runtime for both image and video, so serving any Nemotron-Omni
> visual input needs the same CuTe DSL GEMM build as the Talker MLP above — a
> default build compiles but the visual runner fails to load without it.

## Sampling Parameters

| Parameter | Default | Description |
|---|---:|---|
| `temperature` | `0.7` | Sampling temperature |
| `top_p` | `0.9` | Nucleus sampling threshold |
| `top_k` | `50` | Top-K sampling |
| `logit_bias` | `{}` | Sparse OpenAI-compatible map from token ID to bias value; incompatible with active speculative decoding |
| `max_tokens` | `2048` | Maximum generated tokens |
| `enable_thinking` | `False` | Enables Qwen-style thinking output |
| `disable_spec_decode` | `False` | Disables EAGLE for one request |

Requests with a non-empty `logit_bias` map are rejected while speculative decoding is active. Set
`disable_spec_decode: true` to explicitly use vanilla decoding for that request batch.

## Server-Side Batching

Non-streaming HTTP requests can be micro-batched before entering the runtime:

```bash
python -m experimental.server \
  --model Qwen/Qwen3-1.7B \
  --max-batch-size 16 \
  --enable-batching \
  --max-queue-batch-size 16 \
  --batch-timeout-ms 10
```

Batching is off by default. When enabled, the server groups compatible
non-streaming requests for up to `batch-timeout-ms` milliseconds, then submits
one runtime batch. Requests are compatible when their runtime generation
settings match, including `temperature`, `top_p`, `top_k`, `max_tokens`,
`enable_thinking`, and chat-template settings. Streaming requests bypass the
batcher.

## Request Admission and Backpressure

The server bounds the number of concurrently admitted requests (queued plus
running) with `--request-queue-size` (default `32`):

```bash
python -m experimental.server \
  --model Qwen/Qwen3-1.7B \
  --request-queue-size 32
```

When the queue is full, the server sheds load with a retryable backpressure
status rather than blocking — `503` on the OpenAI endpoints and
`529 overloaded_error` on the Anthropic endpoint, both with `Retry-After: 1`.
Non-streaming requests are admitted into the batcher; streaming requests take a
single runtime slot. Admission never blocks a server worker thread, so a burst
fails fast instead of starving in-flight streaming responses.

When batching is enabled, keep `--request-queue-size` below the server thread
pool (default `40`); a very large queue lets admitted requests occupy every
worker and stall streaming responses.

## Tool Calls

The OpenAI-compatible server accepts `tools`, `tool_choice`,
`assistant.tool_calls`, and `tool` messages. Tool-aware requests are formatted
with the model's Hugging Face chat template before they are sent to the runtime.

`tool_choice` supports `auto`, `none`, `required`, and forced function choices.
Malformed tools, unknown forced tools, and dangling `tool_call_id` values return
a 400 response.

When the model returns a supported tool-call format, non-streaming responses
include `message.tool_calls` and `finish_reason: "tool_calls"`. Streaming
responses include `delta.tool_calls` chunks.

## Anthropic Messages API

The server also exposes a native Anthropic Messages API, so agents that speak
the Anthropic protocol (for example, Claude Code) can target the server
directly by pointing `ANTHROPIC_BASE_URL` at it, with no translation proxy. It
is a thin adapter over the same runtime pipeline as `/v1/chat/completions`.

```bash
curl -sN http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

`system` prompts, `text`/`tool_use`/`tool_result` content blocks, `tools`,
`tool_choice` (`auto`, `any`, `tool`, `none`), and `stop_sequences` are
translated to their OpenAI-pipeline equivalents. Responses carry Anthropic
content blocks, `stop_reason`, and usage. Set `"stream": true` for the Anthropic
SSE event sequence (`message_start`, `content_block_start`/`_delta`/`_stop`,
`message_delta`, `message_stop`).

`POST /v1/messages/count_tokens` returns an input-token estimate for the given
`messages` and `tools` without running generation. It is implemented because
some clients probe it for context management.

### Connecting Claude Code

Point Claude Code at the server with `ANTHROPIC_BASE_URL` — set it to the host
only, as Claude Code appends `/v1/messages` itself. The server does not check
credentials, but the client still requires a non-empty token, so set a dummy
key. Claude Code also requests distinct model tiers internally, so map each
tier to the served model (its id is shown by `GET /v1/models`) to avoid
unknown-model errors:

```bash
ANTHROPIC_BASE_URL=http://localhost:8000 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_OPUS_MODEL=local \
ANTHROPIC_DEFAULT_SONNET_MODEL=local \
ANTHROPIC_DEFAULT_HAIKU_MODEL=local \
claude
```

The same values can be placed in the `env` block of `~/.claude/settings.json`.
Any other Anthropic-Messages-compatible client connects the same way: set its
Anthropic base URL (or provider configuration) to the server's address.

The adapter targets text agentic workloads. Image and document blocks are
dropped, server-executed tools (for example, `web_search`) are not offered to
the model, and a matched custom stop sequence is reported as `end_turn`. On
overload the endpoint returns `529` (see
[Request Admission and Backpressure](#request-admission-and-backpressure)).

## Connecting OpenClaw

OpenClaw speaks the OpenAI protocol, so it connects to `/v1/chat/completions`
rather than the Anthropic API. Register the server as a provider in
`~/.openclaw/openclaw.json` and route an agent to it:

```json
{
  "models": {
    "mode": "merge",
    "providers": {
      "edgellm": {
        "baseUrl": "http://127.0.0.1:8000/v1",
        "apiKey": "dummy",
        "api": "openai-completions",
        "models": [
          {
            "id": "qwen3-8b-edgellm",
            "name": "Qwen3-8B (Edge-LLM)",
            "contextWindow": 32768,
            "maxTokens": 4096,
            "input": ["text"]
          }
        ]
      }
    }
  },
  "agents": {"defaults": {"model": {"primary": "edgellm/qwen3-8b-edgellm"}}}
}
```

The `baseUrl` includes the `/v1` suffix, and `apiKey` may be any non-empty
string. Serve an engine whose `--max-input-len` covers the agent's system
prompt and tool schemas — OpenClaw's context runs to tens of thousands of
tokens, so build a large-context engine rather than a small batching engine.

## EAGLE

```python
from experimental.server import LLM, SamplingParams

llm = LLM(
    engine_dir="/path/to/base/engine",
    eagle_engine_dir="/path/to/eagle/engines",
    draft_top_k=10,
    draft_step=6,
    verify_tree_size=60,
)

outputs = llm.generate(
    ["Explain quantum computing."],
    SamplingParams(max_tokens=256),
)
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/v1/models` | List models |
| `POST` | `/v1/chat/completions` | Chat completions with optional SSE streaming |
| `POST` | `/v1/messages` | Anthropic Messages API with optional SSE streaming |
| `POST` | `/v1/messages/count_tokens` | Anthropic input-token estimate |
| `POST` | `/v1/audio/speech` | Text-to-speech (streamed PCM or WAV) |
| `GET` | `/v1/voices` | Speaker names accepted as `voice` |
| `POST` | `/v1/completions` | Legacy raw-prompt completion (no chat template); streaming or non-stream |
| `POST` | `/v1/audio/transcriptions` | ASR transcription (audio upload); staged errors `400/413/503/500` |

## Notes

- Standard chat templates are applied in the C++ runtime. Tool-aware requests
  are formatted in Python with the model's Hugging Face chat template.
- Thinking output is returned in `reasoning`; final answer text is returned in `content`.
- Supported finish reasons are `stop`, `length`, `cancelled`, and `error`.
- `/v1/chat/completions` accepts `max_completion_tokens` as an alias for
  `max_tokens`; the modern field wins when both are present. The requested
  length is capped at `131072` (a larger value returns a 400).
- On a streaming request, set `stream_options.include_usage: true` to receive a
  final chunk carrying `usage` (empty `choices`) just before `[DONE]`.
- `usage.prompt_tokens` is the runtime's templated, media-expanded prompt
  length when available; otherwise it falls back to the Hugging Face tokenizer
  estimate, which counts each media placeholder once.
