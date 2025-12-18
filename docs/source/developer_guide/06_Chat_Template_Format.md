# Chat Template Format Guide

This document explains how to create and customize chat templates for TensorRT Edge-LLM models.

## Table of Contents
- [Overview](#overview)
- [Chat Template File Format](#chat-template-file-format)
- [Using Chat Templates During Export](#using-chat-templates-during-export)
- [Pre-built Templates](#pre-built-templates)
- [Examples](#examples)
- [System Prompt Priority](#system-prompt-priority)

---

## Overview

Chat templates define how conversational messages are formatted before being processed by the language model. Each model typically has its own chat template format with specific tokens and structures. The chat template ensures that system prompts, user messages, and assistant responses are properly formatted according to the model's training format.

To keep TensorRT Edge-LLM lightweight and free from Jinja dependencies, we use a simple JSON-based chat template format. In practice, the dynamic nature of Jinja templates is only useful when dealing with variables that change the behavior of chat template application, or to handle many different types of messages in one logic block. By restricting messages to follow a specified order and format, and supported uses cases, we can condense the chat template into a list of prefix/suffix pairs and formats for a supported list of multimodalities, achieving the same result with greater simplicity. 

### Key Concepts

- **Roles**: Different message types (system, user, assistant)
- **Prefixes/Suffixes**: Special tokens that wrap each message
- **Content Types**: Formatting for multimodal content (text, image, video)
- **Generation Prompt**: Token sequence that prompts the model to generate a response
- **Default System Prompt**: Fallback system instruction when none is provided

---

## Chat Template File Format

The chat template is defined in a JSON file. When exported with the model, this file is renamed to `processed_chat_template.json` and placed in the model's engine directory.

### File Structure

```json
{
  "model_path": "string (optional)",
  "roles": {
    "system": {
      "prefix": "string",
      "suffix": "string"
    },
    "user": {
      "prefix": "string",
      "suffix": "string"
    },
    "assistant": {
      "prefix": "string",
      "suffix": "string"
    }
  },
  "content_types": {
    "image": {
      "format": "string"
    },
    "video": {
      "format": "string"
    }
  },
  "generation_prompt": "string",
  "default_system_prompt": "string"
}
```

### Field Descriptions

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `roles` | object | Maps role names to their prefix/suffix formatting |
| `roles.system` | object | Formatting for system messages |
| `roles.user` | object | Formatting for user messages |
| `roles.assistant` | object | Formatting for assistant messages |
| `roles.<role>.prefix` | string | Token(s) placed before the message content (can be empty string) |
| `roles.<role>.suffix` | string | Token(s) placed after the message content (can be empty string) |

#### Optional Fields

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `model_path` | string | Path or identifier for the model | "" |
| `content_types` | object | Formatting for multimodal content (images, videos) | {} |
| `content_types.image.format` | string | Token sequence that replaces image content in the formatted prompt | "" |
| `content_types.video.format` | string | Token sequence that replaces video content in the formatted prompt | "" |
| `generation_prompt` | string | Token sequence to prompt model generation | "" |
| `default_system_prompt` | string | Default system instruction when none provided | "" |

---

## Using Chat Templates During Export

By default, when you export a model, the tool:

1. Tries to detect and extract a chat template from the model's tokenizer.
2. If the model is known to have an incompatible or missing template, automatically falls back to a pre-built template (see [Pre-built Templates](#pre-built-templates)).


### Providing a Custom Template (Optional)

You can override the default behavior by supplying your own chat template with the `--chat-template` flag:

```bash
tensorrt-edgellm-export-llm \
    --model_dir /path/to/model \
    --output_dir /path/to/output \
    --chat-template /path/to/my_custom_template.json
```

Provide a custom template when:

1. **Automatic extraction fails**: The export tool cannot reliably detect or convert the model's chat template.
2. **Model is not covered by pre-built templates** but still has issues with its tokenizer template.
3. **You want different formatting** than the model's default (for example, different system prompts, role tokens, or multimodal placeholders).

---

## Pre-built Templates

Some models have known issues with their tokenizer's chat template definitions (e.g., missing templates or incompatible formats). For these models, TensorRT Edge-LLM includes pre-built templates that are used automatically.

**Location:** `tensorrt_edgellm/chat_templates/templates/`

| Model | Template File | Reason |
|-------|--------------|--------|
| Phi-4-Multimodal | `phi4mm.json` | Tokenizer lacks proper multimodal chat template definition |

**Automatic Fallback:**
When you export a model that is on the incompatible list (like Phi-4-Multimodal), the system automatically uses the pre-built template. You don't need to manually specify it unless you want to override it.

---

## Examples

### Basic Text-Only Model

```json
{
  "model_path": "/path/to/Qwen2-7B",
  "roles": {
    "system": {
      "prefix": "<|im_start|>system\n",
      "suffix": "<|im_end|>\n"
    },
    "user": {
      "prefix": "<|im_start|>user\n",
      "suffix": "<|im_end|>\n"
    },
    "assistant": {
      "prefix": "<|im_start|>assistant\n",
      "suffix": "<|im_end|>\n"
    }
  },
  "content_types": {},
  "generation_prompt": "<|im_start|>assistant\n",
  "default_system_prompt": "You are a helpful assistant"
}
```

### Multimodal Model with Content Types

For vision-language models, the `format` field specifies the token sequence that replaces each image or video in the formatted prompt:

```json
{
  "model_path": "/path/to/Qwen2-VL-7B",
  "roles": {
    "system": {
      "prefix": "<|im_start|>system\n",
      "suffix": "<|im_end|>\n"
    },
    "user": {
      "prefix": "<|im_start|>user\n",
      "suffix": "<|im_end|>\n"
    },
    "assistant": {
      "prefix": "<|im_start|>assistant\n",
      "suffix": "<|im_end|>\n"
    }
  },
  "content_types": {
    "image": {
      "format": "<|vision_start|><|image_pad|><|vision_end|>"
    },
    "video": {
      "format": "<|vision_start|><|video_pad|><|vision_end|>"
    }
  },
  "generation_prompt": "<|im_start|>assistant\n",
  "default_system_prompt": "You are a helpful assistant."
}
```

### Multi-turn Conversation Example

**Template:**
```json
{
  "roles": {
    "system": {
      "prefix": "<|im_start|>system\n",
      "suffix": "<|im_end|>\n"
    },
    "user": {
      "prefix": "<|im_start|>user\n",
      "suffix": "<|im_end|>\n"
    },
    "assistant": {
      "prefix": "<|im_start|>assistant\n",
      "suffix": "<|im_end|>\n"
    }
  },
  "generation_prompt": "<|im_start|>assistant\n",
  "default_system_prompt": "You are a helpful assistant"
}
```

And this input:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a math tutor."
    },
    {
      "role": "user",
      "content": "What is 2+2?"
    },
    {
      "role": "assistant",
      "content": "2+2 equals 4."
    },
    {
      "role": "user",
      "content": "What about 3+3?"
    }
  ]
}
```

The formatted output will be:
```
<|im_start|>system
You are a math tutor.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
2+2 equals 4.<|im_end|>
<|im_start|>user
What about 3+3?<|im_end|>
<|im_start|>assistant

```
---

## System Prompt Priority

The system prompt is determined by the following priority order:

1. **Explicit system message in request** (highest priority)
2. **`default_system_prompt` from input JSON**
3. **`default_system_prompt` from chat template** (lowest priority)

If none of these are provided, no system prompt is added to the conversation.

**Example scenarios:**

**Scenario 1: Explicit system message (highest priority)**
```json
{
  "messages": [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is 2+2?"}
  ]
}
```
Result: Uses "You are a math tutor." regardless of other defaults.

**Scenario 2: Using input JSON default**
```json
{
  "default_system_prompt": "You are a helpful coding assistant.",
  "requests": [
    {
      "messages": [
        {"role": "user", "content": "Explain Python decorators."}
      ]
    }
  ]
}
```
Result: Since no explicit system message exists, uses "You are a helpful coding assistant."

**Scenario 3: Fallback to chat template default**
```json
{
  "requests": [
    {
      "messages": [
        {"role": "user", "content": "Hello!"}
      ]
    }
  ]
}
```
Result: Uses the `default_system_prompt` from `processed_chat_template.json` (if defined).

For more details on the input JSON format, see [INPUT_FORMAT.md](../../../examples/llm/INPUT_FORMAT.md).