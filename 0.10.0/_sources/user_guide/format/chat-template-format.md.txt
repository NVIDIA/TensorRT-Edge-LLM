# Chat Template Format

Chat templates define how conversational messages are formatted for the language model. This guide explains the JSON-based chat template format used by TensorRT Edge-LLM.

## Overview

Our implementation follows HuggingFace's `apply_chat_template` API, but uses a lightweight JSON format instead of Jinja templates. The chat template is automatically extracted during model export and saved as `processed_chat_template.json`.

## File Structure

```json
{
  "roles": {
    "system": {"prefix": "string", "suffix": "string"},
    "user": {"prefix": "string", "suffix": "string"},
    "assistant": {"prefix": "string", "suffix": "string"}
  },
  "content_types": {
    "image": {"format": "string"},
    "video": {"format": "string"}
  },
  "generation_prompt": "string",
  "generation_prompt_thinking": "string (optional)",
  "default_system_prompt": "string"
}
```

### Fields
- **`roles`** (required): Prefix/suffix tokens for each role
- **`content_types`** (optional): Format tokens for images/videos
- **`generation_prompt`** (optional): Token sequence to start generation
- **`generation_prompt_thinking`** (optional): Alternative prompt for thinking mode
- **`default_system_prompt`** (optional): Default system instruction

## Thinking Mode

Some models (like Qwen3) support "thinking mode" where the model generates its reasoning process. This is controlled by `enable_thinking` in the input JSON.

- **`enable_thinking: false`** (default): Uses `generation_prompt`
- **`enable_thinking: true`**: Uses `generation_prompt_thinking`

The system automatically detects thinking mode support during model export. If not supported, the parameter has no effect.

## Custom Templates

`tensorrt_edgellm` extracts the chat template from the checkpoint during export and
writes `processed_chat_template.json` next to the exported LLM ONNX graph.

To customize the chat template, first export the model normally, then edit the
generated file in place:

```bash
# 1. Export (generates processed_chat_template.json with auto-detected template)
tensorrt-edgellm-export \
    /path/to/model \
    /path/to/output

# 2. Edit the generated template to customize roles, delimiters, etc.
vi /path/to/output/llm/processed_chat_template.json
```

The generated `processed_chat_template.json` is a structured JSON with the
following schema (not a raw Jinja template):

```json
{
  "model_path": "path/to/model",
  "roles": {
    "system":    {"prefix": "<prefix>", "suffix": "<suffix>"},
    "user":      {"prefix": "<prefix>", "suffix": "<suffix>"},
    "assistant": {"prefix": "<prefix>", "suffix": "<suffix>"}
  },
  "content_types": {
    "image": {"format": "<placeholder>"},
    "video": {"format": "<placeholder>"},
    "audio": {"format": "<placeholder>"}
  },
  "generation_prompt": "<prompt>",
  "default_system_prompt": "<system_message>"
}
```

Reference templates are available under `tensorrt_edgellm/chat_templates/`
(e.g., `phi4mm.json`, `qwen3asr.json`, `nemotron_nano_v2.json`).

Pre-built templates are automatically used for models with known tokenizer issues (e.g., Phi-4-Multimodal).

## Examples

### Basic Text-Only Model (Qwen2)

```json
{
  "roles": {
    "system": {"prefix": "<|im_start|>system\n", "suffix": "<|im_end|>\n"},
    "user": {"prefix": "<|im_start|>user\n", "suffix": "<|im_end|>\n"},
    "assistant": {"prefix": "<|im_start|>assistant\n", "suffix": "<|im_end|>\n"}
  },
  "generation_prompt": "<|im_start|>assistant\n",
  "default_system_prompt": "You are a helpful assistant"
}
```

### Multimodal Model (Qwen2-VL)

```json
{
  "roles": {
    "system": {"prefix": "<|im_start|>system\n", "suffix": "<|im_end|>\n"},
    "user": {"prefix": "<|im_start|>user\n", "suffix": "<|im_end|>\n"},
    "assistant": {"prefix": "<|im_start|>assistant\n", "suffix": "<|im_end|>\n"}
  },
  "content_types": {
    "image": {"format": "<|vision_start|><|image_pad|><|vision_end|>"}
  },
  "generation_prompt": "<|im_start|>assistant\n",
  "default_system_prompt": "You are a helpful assistant."
}
```

### Model with Thinking Mode (Qwen3)

```json
{
  "roles": {
    "system": {"prefix": "<|im_start|>system\n", "suffix": "<|im_end|>\n"},
    "user": {"prefix": "<|im_start|>user\n", "suffix": "<|im_end|>\n"},
    "assistant": {"prefix": "<|im_start|>assistant\n", "suffix": "<|im_end|>\n"}
  },
  "generation_prompt": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
  "generation_prompt_thinking": "<|im_start|>assistant\n",
  "default_system_prompt": "You are a helpful assistant"
}
```

## System Prompt Priority

1. Explicit system message in request (highest)
2. `default_system_prompt` from chat template
3. No system prompt if neither provided
