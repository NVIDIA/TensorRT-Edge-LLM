# Input JSON Format

This guide describes the input JSON format for the LLM inference tool. The format supports text-only and multimodal inputs, multi-turn conversations, LoRA adapters, and advanced features.

## Basic Structure

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 50,
    "logit_bias": {"123": -100.0},
    "max_generate_length": 256,
    "num_logprobs": 0,
    "apply_chat_template": true,
    "enable_thinking": false,
    "available_lora_weights": {},
    "requests": [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Your message here"
                }
            ],
            "lora_name": "optional_lora_name",
            "save_system_prompt_kv_cache": false,
            "disable_spec_decode": false,
            "logit_bias": {"456": 5.0},
            "num_logprobs": 0,
            "stop": ["optional_stop_string"]
        }
    ]
}
```


## Parameters

### Required
- **`requests`**: Array of conversation requests

### Optional Global Parameters
- **`batch_size`** (default: 1): Number of requests per batch
- **`temperature`** (default: 1.0): Sampling temperature (0.0 = deterministic)
- **`top_p`** (default: 0.8): Nucleus sampling threshold
- **`top_k`** (default: 50): Top-k sampling parameter
- **`logit_bias`** (optional): Sparse map from token ID to bias value. The top-level map is the default for all requests.
- **`max_generate_length`** (default: 256): Maximum tokens to generate
- **`apply_chat_template`** (default: true): Apply chat template formatting
- **`add_generation_prompt`** (default: true): Add generation prompt token sequence
- **`enable_thinking`** (default: false): Enable thinking mode (Qwen3+)
- **`num_logprobs`** (default: 0): Number of top log-probabilities to return per generated token. `0` disables logprobs. Maximum value is 50; values outside `[0, 50]` are rejected at input parsing. The cap applies to the per-token K of each individual request — it is **not** a cumulative budget across requests or batches: a batch simply computes at the maximum K its requests asked for, and every request in that batch receives that K. Each candidate carries `token_id`, `token` (decoded string), `bytes` (raw token bytes) and `logprob`, following OpenAI API semantics (`log(softmax(logits))`). The top-level value is the default for all requests; a request may raise it. Supported for both vanilla and EAGLE speculative decoding.
- **`available_lora_weights`** (default: {}): Map of LoRA adapter names to file paths

### Request Fields
- **`messages`** (required): Array of conversation messages
- **`lora_name`** (optional): LoRA adapter name from `available_lora_weights`
- **`save_system_prompt_kv_cache`** (optional): Cache system prompt KV for reuse
- **`disable_spec_decode`** (optional, default: false): Disable EAGLE speculative decoding for this request even if draft engine is loaded
- **`logit_bias`** (optional): Request-specific sparse logit-bias map. When set, it overrides the top-level `logit_bias` default for this request.
- **`num_logprobs`** (optional): Overrides the top-level `num_logprobs` default for this request. Applied batch-uniformly (like `disable_spec_decode`): the batch computes at the maximum value requested by any request in it.
- **`stop`** (optional): String or array of strings that halt generation when produced in the output. The stop string itself is excluded from the returned text. Each request in a batch may declare its own list independently. Defaults to no stop strings.

### Message Fields
- **`role`**: `"system"`, `"user"`, or `"assistant"`
- **`content`**: String (text-only) or Array (multimodal)

**Content Array Format:**
- Text: `{"type": "text", "text": "..."}`
- Image: `{"type": "image", "image": "/path/to/image.jpg"}`
- Audio: `{"type": "audio", "audio": "/path/to/clip.wav"}` (raw `.wav` / `.mp3` / `.flac` decoded in C++ via vendored miniaudio + in-tree mel extractor. Feature-extractor family — `whisper` / `parakeet` — is auto-derived from the engine's `audio/config.json`, mirroring HF / vLLM where FE is pinned by the model. The HTTP server in `experimental.server` accepts the same audio formats via `input_audio` / `audio_url` and routes through the same C++ mel path.)
- Video: `{"type": "video", "video": "/path/to/video.mp4"}` *(Note: Video support is a placeholder for future releases and is not available for now)*

## Examples

### Basic Text Input

```json
{
    "batch_size": 1,
    "max_generate_length": 256,
    "requests": [
        {
            "messages": [
                {"role": "user", "content": "What is machine learning?"}
            ]
        }
    ]
}
```

### Multi-Turn Conversation

```json
{
    "requests": [
        {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."},
                {"role": "user", "content": "What is the population?"}
            ]
        }
    ]
}
```

### Multimodal Input (Vision-Language Models)

```json
{
    "requests": [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "/path/to/image.jpg"},
                        {"type": "text", "text": "Describe this image."}
                    ]
                }
            ]
        }
    ]
}
```

### LoRA Adapters

```json
{
    "available_lora_weights": {
        "french": "/path/to/french_adapter.safetensors"
    },
    "requests": [
        {
            "messages": [
                {"role": "user", "content": "Translate to French."}
            ],
            "lora_name": "french"
        }
    ]
}
```

**Note:** All requests in the same batch must use the same LoRA adapter.

### System Prompt KV Cache

```json
{
    "requests": [
        {
            "messages": [
                {"role": "system", "content": "Long system prompt..."},
                {"role": "user", "content": "Question?"}
            ],
            "save_system_prompt_kv_cache": true
        }
    ]
}
```

### Raw Format (No Chat Template)

```json
{
    "apply_chat_template": false,
    "requests": [
        {
            "messages": [
                {"role": "user", "content": "Raw text without special tokens"}
            ]
        }
    ]
}
```

### Disable Speculative Decoding

When using EAGLE speculative decoding, you can disable it for specific requests:

```json
{
    "requests": [
        {
            "messages": [
                {"role": "user", "content": "Your question here"}
            ],
            "disable_spec_decode": true
        }
    ]
}
```

**Use cases:**
- Quality: Some inputs may benefit from standard decoding over EAGLE
- Switching strategies: Different batches can use different decoding strategies (one batch with EAGLE, another without)
- Debugging: Compare performance with/without speculative decoding

**Note:** If any request in a batch has `disable_spec_decode: true`, speculative decoding will be disabled for the entire batch. Requests within one batch cannot use different decoding strategies simultaneously for now.

### Logit Bias

`logit_bias` accepts a sparse map of tokenizer token IDs to additive logit bias values. Positive values make a token more likely, and negative values make it less likely. Bias values must be finite and in `[-100.0, 100.0]`; each map may contain up to 1024 token IDs.

Top-level `logit_bias` applies to every request by default. A request-level `logit_bias` overrides the top-level default for that request.

```json
{
    "logit_bias": {"123": -100.0},
    "requests": [
        {
            "messages": [
                {"role": "user", "content": "Avoid token 123 by default."}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Prefer token 456 for this request."}
            ],
            "logit_bias": {"456": 5.0}
        }
    ]
}
```

**Speculative decoding limitation:** Requests with a non-empty `logit_bias` map are rejected while speculative decoding is active. Set `disable_spec_decode: true` to explicitly use vanilla decoding for that batch before sending logit bias.

### Top-N Log-Probabilities

Return the top-5 most likely tokens (with log-probabilities) at each generation step:

```json
{
    "num_logprobs": 5,
    "max_generate_length": 64,
    "requests": [
        {
            "messages": [
                {"role": "user", "content": "The capital of France is"}
            ]
        }
    ]
}
```

`num_logprobs` may also be set per request to override the top-level default (see Request Fields).

The output JSON will contain a `logprobs` field alongside `output_text`:

```json
"logprobs": [
    [
        {"token_id": 3681, "token": " Paris", "bytes": [32, 80, 97, 114, 105, 115], "logprob": -0.041},
        {"token_id": 8098, "token": " paris", "bytes": [32, 112, 97, 114, 105, 115], "logprob": -3.812},
        {"token_id":  627, "token": ".",       "bytes": [46],                        "logprob": -4.201},
        {"token_id": 4892, "token": " PARIS", "bytes": [32, 80, 65, 82, 73, 83],     "logprob": -5.103},
        {"token_id": 3085, "token": " France","bytes": [32, 70, 114, 97, 110, 99, 101], "logprob": -5.447}
    ],
    ...
]
```

Each element of the outer array corresponds to one generated token (step); the inner array lists up to `num_logprobs` candidates sorted by descending probability. Each candidate carries `token_id`, `token`, `bytes` and `logprob`.

`token` is the token decoded as UTF-8 with invalid/partial bytes replaced by U+FFFD; `bytes` is the raw token bytes. For byte-level BPE tokenizers a single token may be only part of a multi-byte character (e.g. half a CJK character), so `token` can be lossy — concatenate `bytes` across tokens to reconstruct the exact text losslessly.

**Notes:**
- `num_logprobs` can be set top-level (default for all requests) and/or per request (override); it is batch-uniform — the batch computes at the maximum value requested, and all requests in the batch share the same K.
- Logprobs are computed at temperature=1.0 regardless of the `temperature` sampling setting.

### Stop Strings

Generation halts as soon as any of the specified substrings appears in the decoded output; the stop string itself is excluded from the returned text. Accepts a single string or an array. Each request carries its own independent list — requests in the same batch may stop on different strings or none at all.

```json
{
    "requests": [
        {
            "messages": [
                {"role": "user", "content": "Write a short answer ending before '###'."}
            ],
            "stop": ["###", "\n\nUser:"]
        }
    ]
}
```

When a stop string triggers termination, the request's finish reason is `stop-words`. Earliest position in the decoded output wins when multiple stop strings could match.

## Notes

- System prompt: Uses provided system message, or model default from chat template
- LoRA: All requests in same batch must use same adapter
- Logit bias: Supported through vanilla decoding; active speculative decoding must be explicitly disabled first.
- Paths: Use absolute or relative paths for images/videos
- Format: Follows OpenAI chat completion API structure
