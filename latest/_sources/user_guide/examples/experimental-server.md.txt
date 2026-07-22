# Experimental High-Level Python API and Server

The experimental Python API wraps export, engine build, engine loading, generation, streaming, and OpenAI-compatible serving.

> **Status:** Experimental. API may change between releases.

## Prerequisites

Complete the [Installation Guide](../getting_started/installation.md) with the C++ runtime, Python bindings, and server dependencies enabled before proceeding. The examples below assume `experimental.server` and `tensorrt_edgellm` are importable from the active Python environment.

If the active environment was installed with base export dependencies only, install the server dependencies before building Python bindings or launching the server:

```bash
cd /path/to/TensorRT-Edge-LLM
pip install -r requirements-server.txt
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

For VLMs and models that support audio (Qwen3-Omni, Qwen3-ASR, Nemotron-Omni), also pass `visual_onnx_dir` or `visual_engine_dir`.

## Audio Input

The server accepts three OpenAI-compatible content forms inside user
messages for models that support audio (Qwen3-Omni, Qwen3-ASR, Nemotron-Omni):

```json
{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}
{"type": "audio_url", "audio_url": {"url": "file:///abs/path | data:audio/...;base64,..."}}
{"type": "audio", "audio": "<local path>"}
```

`http(s)://` URLs are rejected by design — host the audio locally and use
`file://`, or inline the bytes as base64 via `input_audio`. Supported
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

## Notes

- Standard chat templates are applied in the C++ runtime. Tool-aware requests
  are formatted in Python with the model's Hugging Face chat template.
- Thinking output is returned in `reasoning`; final answer text is returned in `content`.
- Supported finish reasons are `stop`, `length`, `cancelled`, and `error`.
