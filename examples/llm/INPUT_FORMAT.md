# Input JSON File Format

This document describes the required format for the input JSON file used with the LLM inference tool.

## Overview

The input JSON file contains configuration parameters and a list of requests to be processed by the LLM. Each request is a conversation consisting of multiple messages with roles and content. The tool supports both text-only and multimodal (text + images) inputs, as well as multi-turn conversations.

## File Structure

The JSON file must contain the following top-level structure:

```json
{
    "batch_size": <integer>,
    "temperature": <float>,
    "top_p": <float>,
    "top_k": <integer>,
    "max_generate_length": <integer>,
    "apply_chat_template": <boolean>,  // optional (default: true)
    "enable_thinking": <boolean>,  // optional (default: false, Qwen3-specific)
    "available_lora_weights": {  // optional. Only needed for LoRA engines.
        "<lora_name>": "<path_to_safetensors_file>",
        ...
    },
    "requests": [
        {
            "messages": [
                {
                    "role": "<string>",  // "system", "user", or "assistant"
                    "content": "<string>"  // Simple string for text-only messages
                    // OR
                    "content": [  // Array format for multimodal content
                        {
                            "type": "<string>",  // "text", "image", or "video"
                            "text": "<string>",  // for type="text"
                            "image": "<path>",   // for type="image"
                            "video": "<path>"    // for type="video"
                        }
                    ]
                }
            ],
            "lora_name": "<string>",  // optional. Name of LoRA weights from available_lora_weights.
            "save_system_prompt_kv_cache": <boolean>  // optional (default: false). Save system prompt KV cache for reuse.
        }
    ]
}
```


## LoRA (Low-Rank Adaptation) Support

LoRA enables fine-tuned model inference using adapter weights. 

### Defining Available LoRA Weights
First, define all available LoRA adapters in the global `available_lora_weights` map:
```json
{
    "available_lora_weights": {
        "french_adapter": "/path/to/french_adapter.safetensors",
        "spanish_adapter": "/path/to/spanish_adapter.safetensors",
        "jailbreak_detector": "/path/to/jailbreak_detector.safetensors"
    }
}
```

### Per-Conversation LoRA Selection
Then reference these adapters by name in each request:
```json
{
    "available_lora_weights": {
        "french_adapter": "/path/to/french_adapter.safetensors",
        "spanish_adapter": "/path/to/spanish_adapter.safetensors"
    },
    "requests": [
        {
            "messages": [...],
            "lora_name": "french_adapter"
        },
        {
            "messages": [...],
            "lora_name": "spanish_adapter"
        }
    ]
}
```

### Requirements:
- TensorRT engine built with LoRA support
- LoRA weights in `.safetensors` format
- LoRA adapters must be defined in `available_lora_weights` before being referenced by `lora_name`
- **Important:** All requests within the same batch must use the same LoRA weights. Different LoRA weights are only supported across different batches.

## Global Parameters

### Required Parameters

- **`requests`** (array of objects): A list of conversation requests. Each request is an object containing a `messages` array and optional per-conversation configuration.

### Optional Parameters

- **`batch_size`** (integer, default: 1): Number of requests to process in a single batch
- **`temperature`** (float, default: 1.0): Controls randomness in generation (0.0 = deterministic, higher = more random)
- **`top_p`** (float, default: 0.8): Nucleus sampling parameter (0.0-1.0)
- **`top_k`** (integer, default: 50): Top-k sampling parameter
- **`max_generate_length`** (integer, default: 256): Maximum number of tokens to generate
- **`apply_chat_template`** (boolean, default: true): Whether to apply chat template formatting with special tokens. When set to `false`, messages will be concatenated without role prefixes/suffixes or special tokens, useful for models that don't require chat template formatting
- **`enable_thinking`** (boolean, default: false): Whether to enable thinking mode for models that support it. When set to `false`, standard generation prompt is used. When set to `true`, thinking-enabled generation prompt is used if available. This parameter only affects models with thinking mode support and is ignored for other models
- **`available_lora_weights`** (object, default: {}): Map of LoRA adapter names to their file paths. Only needed for LoRA-enabled engines

## System Prompt Behavior

- If a system message is provided in the request, it will be used
- If no system message is provided, the model's default system prompt from the chat template will be used (if available)

## Request Structure

Each request in the `requests` array is an object with the following fields:

### Required Fields

- **`messages`** (array): An array of messages that form a conversation. This enables multi-turn conversations with context from previous exchanges.

### Optional Fields

- **`lora_name`** (string): Name of the LoRA adapter to use for this conversation, referencing an entry in the global `available_lora_weights` map. This allows different conversations to use different fine-tuned adapters. Note that all requests within the same batch must use the same LoRA weights.

- **`save_system_prompt_kv_cache`** (boolean, default: false): Whether to save the system prompt KV cache for later reuse. This is useful for optimizing performance when using the same system prompt across multiple requests, as it avoids recomputing the KV cache for the system prompt. Note: If any request in a batch sets this to `true`, all requests in that batch will cache the system prompt KV cache.

### Message Structure

Each message object contains:

#### Required Fields

- **`role`** (string): The role of the message sender. Must be one of:
  - `"system"`: System instructions or context (optional, model default will be used if not provided)
  - `"user"`: User input or question
  - `"assistant"`: Assistant's previous response (for multi-turn conversations)

- **`content`** (string or array): The message content. Can be:
  - **String format** (text-only, simpler): Direct text string
  - **Array format** (multimodal): Array of content items for text, images, videos

#### Content Formats

**Simple String Format (Text-Only Messages):**

For text-only messages, you can use a simple string:
```json
"content": "Your text message here"
```

**Array Format (Multimodal Messages):**

For messages with images, videos, or mixed content, use an array of content items.

Each content item has a `type` field and type-specific fields:

**For text content:**
- **`type`**: `"text"`
- **`text`** (string): The text content

**For image content:**
- **`type`**: `"image"`
- **`image`** (string): Path to the image file

**For video content:**
- **`type`**: `"video"`
- **`video`** (string): Path to the video file

## Examples

### Text-Only Input (Single Request)

Using the simple string format for text-only messages:

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 50,
    "max_generate_length": 256,
    "requests": [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Introduce NVIDIA and introduce the CEO of this company."
                }
            ]
        }
    ]
}
```

**Note:** You can also use the array format for text-only messages if preferred:
```json
"content": [{"type": "text", "text": "Your message here"}]
```

### Multi-Turn Conversation

Using simple string format for easy text conversations:

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 50,
    "max_generate_length": 128,
    "requests": [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "What is the capital of France?"
                },
                {
                    "role": "assistant",
                    "content": "The capital of France is Paris."
                },
                {
                    "role": "user",
                    "content": "What is the population of that city?"
                }
            ]
        }
    ]
}
```

### Multimodal Input (Text + Images)

For multimodal content, use the array format:

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 50,
    "max_generate_length": 256,
    "requests": [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "image1.jpeg"},
                        {"type": "image", "image": "image2.jpeg"},
                        {"type": "text", "text": "Compare these two images and identify the similarities."}
                    ]
                }
            ]
        }
    ]
}
```

### Batch Processing (Multiple Requests)

You can mix string format (text-only) and array format (multimodal) in the same batch:

```json
{
    "batch_size": 2,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 50,
    "max_generate_length": 256,
    "requests": [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "What is machine learning?"
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "diagram.png"},
                        {"type": "text", "text": "Explain this diagram."}
                    ]
                }
            ]
        }
    ]
}
```

### LoRA Input (Per-Conversation)

The format allows you to specify different LoRA adapters for each conversation by referencing them by name:

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 50,
    "max_generate_length": 256,
    "available_lora_weights": {
        "french_adapter": "/path/to/french_adapter.safetensors",
        "spanish_adapter": "/path/to/spanish_adapter.safetensors"
    },
    "requests": [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Translate this text to French."
                }
            ],
            "lora_name": "french_adapter"
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Translate this text to Spanish."
                }
            ],
            "lora_name": "spanish_adapter"
        }
    ]
}
```

**Note:** The above example will naturally process each conversation in separate batches (assuming `batch_size` is 1) since they use different LoRA weights. If you manually group them into the same batch by setting `batch_size` to 2 or higher, **the program will error out** with the message: "Different LoRA weights within the same batch are not supported".

### Raw Format Input (Without Chat Template)

When you want to use raw concatenation without chat template special tokens, set the global `apply_chat_template` parameter to `false`:

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 50,
    "max_generate_length": 256,
    "apply_chat_template": false,
    "requests": [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is the capital of France?"
                }
            ]
        }
    ]
}
```

This will produce a raw prompt without any special tokens like `<|im_start|>`, `<|im_end|>`, etc. The text will be concatenated directly. This is useful for:
- Models trained without chat templates
- Custom prompt engineering
- Direct control over the input format

Note: The `apply_chat_template` flag applies to all requests in the batch for consistency.

### System Prompt KV Cache Optimization

When using long system prompts repeatedly, you can set `save_system_prompt_kv_cache` to `true` in a request to cache the system prompt's KV cache for reuse:

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 50,
    "max_generate_length": 256,
    "requests": [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant with extensive knowledge..."
                },
                {
                    "role": "user",
                    "content": "What is the capital of France?"
                }
            ],
            "save_system_prompt_kv_cache": true
        }
    ]
}
```

This optimization is particularly useful for:
- Long system prompts that are reused across multiple requests
- Initialization phase setup where you want to cache system instructions
- Scenarios where the same system context is used repeatedly

## Processing Behavior

1. **Chat Template Application**: The chat template (loaded from `processed_chat_template.json`) is automatically applied to format messages with the appropriate role prefixes/suffixes and special tokens when `apply_chat_template` is `true` (default)
2. **Raw Format Mode**: When `apply_chat_template` is set to `false`, messages are concatenated without role-specific tokens or special formatting. This is useful for:
   - Models that don't use chat templates
   - Custom prompt formats
   - Direct control over prompt structure
   - Simple concatenation of text and images
3. **System Prompts**: 
   - If a system message is provided, it will be used
   - If no system message is provided, the model's default system prompt from the chat template will be used (if available)
4. **Batching**: Requests are processed in batches according to the `batch_size` parameter
5. **Multi-Turn Support**: Each request can contain multiple messages to support conversation context
6. **Content Type Handling**: 
   - Text content is directly inserted into the formatted output
   - Image/video placeholders are formatted according to the chat template
7. **Image Loading**: Images are loaded from the specified file paths during processing
8. **LoRA Weights**: LoRA adapters are loaded once at initialization from `available_lora_weights`, then switched per batch based on `lora_name` references
9. **System Prompt KV Cache**: When `save_system_prompt_kv_cache` is set to `true` for a request, the system prompt's KV cache is saved for reuse, improving performance for repeated system prompts
10. **Error Handling**: The tool will throw errors if:
   - The JSON file cannot be parsed
   - A message is missing the required `role` or `content` field
   - A request object is missing the required `messages` field
   - The `requests` field is not an array of objects
   - Unknown content types are specified
   - Different LoRA weights are specified for requests within the same batch
   - A `lora_name` is referenced that is not defined in `available_lora_weights`

## Notes

- Image and video paths should be relative to the working directory or absolute paths
- The chat template automatically adds appropriate special tokens and formatting
- Assistant messages in the middle of a conversation enable multi-turn interactions with context
- The format follows OpenAI's chat completion API structure for better interoperability

## Key Design Principles

The input format is designed to:
- **Follow OpenAI's chat completion API structure** for better interoperability
- **Support multi-turn conversations** with full context from previous exchanges
- **Enable per-conversation LoRA weights** for different fine-tuned adapters
- **Handle multimodal inputs** (text, images, videos) in a unified way
- **Maintain clear separation** between conversation requests and global parameters