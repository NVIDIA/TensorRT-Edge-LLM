# System Prompt Cache

## Overview

System (instruction) prompt cache optimizes prefill latency by reusing KV cache from previously computed system prompts. When the same system prompt is used across multiple requests, the runtime skips re-computing its KV cache, reducing time-to-first-token (TTFT).

**Key Points**:
- First request with `save_system_prompt_kv_cache: true` generates and saves the cache
- Subsequent requests automatically reuse the cached KV cache
- Cache is keyed by `(system_prompt_text, lora_weights_name)` — exact string match required
- In-memory only — cache persists for runtime lifetime but not across restarts

---

## Usage

### Basic Example

First request saves the cache:

```json
{
    "requests": [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful Python programming assistant."},
                {"role": "user", "content": "How do I read a CSV file?"}
            ],
            "save_system_prompt_kv_cache": true
        }
    ]
}
```

Subsequent requests automatically reuse it (no flag needed):

```json
{
    "requests": [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful Python programming assistant."},
                {"role": "user", "content": "How do I write JSON?"}
            ]
        }
    ]
}
```

### LoRA Support

Caches are LoRA-aware — different LoRA adapters create separate cache entries for the same system prompt:

```json
{
    "available_lora_weights": {
        "french": "/path/to/french_adapter.safetensors"
    },
    "requests": [
        {
            "messages": [
                {"role": "system", "content": "Translate the following to French."},
                {"role": "user", "content": "Hello"}
            ],
            "lora_name": "french",
            "save_system_prompt_kv_cache": true
        }
    ]
}
```

### EAGLE Speculative Decoding

Fully supported — both base and draft model KV caches are saved and reused automatically. No special configuration needed.

---

## Limitations

- **Multimodal system prompt not supported**: System prompt shall only contain text data.
- **Exact string match required**: Any whitespace or punctuation difference causes cache miss
- **In-memory only**: Cache lost when runtime terminates
- **Chat template aware**: When `apply_chat_template: true` (default), formatted prompt is used as cache key

---

## Best Practices

**When to Use**:
- Long system prompts (> 1K tokens) used repeatedly
- Multi-tenant serving with role-specific system prompts
- Agent systems with consistent instruction templates

**Not Recommended**:
- Short prompts (< 100 tokens) — overhead exceeds benefit
- Frequently changing prompts

**Tips**:
- Standardize system prompts to maximize cache hit rate
- Pre-warm cache during initialization with `save_system_prompt_kv_cache: true`

---

## Notes

- No build-time configuration required — feature is always available
- Enable debug logging (`llm_inference --debug`) to verify cache usage
- Prefill metrics track reused vs. computed tokens for monitoring cache effectiveness
