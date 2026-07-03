# Input JSON Format

This guide describes the input JSON format for the LLM inference tool. The format supports text-only and multimodal inputs, multi-turn conversations, LoRA adapters, and advanced features.

## Basic Structure

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 50,
    "max_generate_length": 256,
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
- **`max_generate_length`** (default: 256): Maximum tokens to generate
- **`apply_chat_template`** (default: true): Apply chat template formatting
- **`add_generation_prompt`** (default: true): Add generation prompt token sequence
- **`enable_thinking`** (default: false): Enable thinking mode (Qwen3+)
- **`available_lora_weights`** (default: {}): Map of LoRA adapter names to file paths

### Request Fields
- **`messages`** (required): Array of conversation messages
- **`lora_name`** (optional): LoRA adapter name from `available_lora_weights`
- **`save_system_prompt_kv_cache`** (optional): Cache system prompt KV for reuse
- **`disable_spec_decode`** (optional, default: false): Disable EAGLE speculative decoding for this request even if draft engine is loaded
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
- Paths: Use absolute or relative paths for images/videos
- Format: Follows OpenAI chat completion API structure
