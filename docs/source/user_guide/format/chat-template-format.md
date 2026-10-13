# Chat Template Format

TensorRT Edge-LLM uses each checkpoint's provider-owned Jinja template. During
export or checkpoint-direct build, the template is copied without rewriting.
The C++ runtime loads `chat_template.jinja` directly with Pantor Inja. It does
not invoke Python, generate a second template representation, or select a
model-family template.

## Exported Artifact

`tensorrt-edgellm-export` and the checkpoint-direct builder copy the provider
template beside the LLM engine artifacts:

```text
llm/
├── chat_template.jinja
├── chat_template.processor # Present for a provider-defined raw processor contract
├── config.json
├── tokenizer.json
└── tokenizer_config.json
```

The authoritative source is the checkpoint's standalone
`chat_template.jinja`, or its `chat_template` field in the provider's
`chat_template.json`, `processor_config.json`, or `tokenizer_config.json`.
These JSON files are checkpoint metadata containers whose field value is
provider-owned Jinja source. Export materializes one default provider template
as `chat_template.jinja` and removes any JSON chat-template artifact from the
runtime directory. Named template sets fail explicitly; supported Edge-LLM
models carry tool behavior in their default provider template.
Edge-LLM neither exports nor loads a JSON chat-template format. Pantor Inja
parses the Jinja files when the runtime loads the model. A provider construct
that Pantor Inja does not support fails model loading instead of being
rewritten or silently changing the prompt.

Following vLLM's server contract, the runtime detects whether the provider
template iterates over message content and normalizes each request to content
blocks or strings before rendering. Tool results are normalized to strings. A
string-only multimodal template requires an explicit model processor contract;
Edge-LLM does not guess media tokens.

Some older multimodal checkpoints publish a string-only template even though
their API accepts structured media. Phi-4MM's raw processor turns media blocks
into numbered placeholders, and legacy InternVL's provider path inserts its
image and video sentinels. `chat_template.processor` records these
provider-owned contracts. The C++ runtime applies them explicitly and rejects
unsupported content types; it does not substitute a generic template.

Models that do not publish Jinja semantics use an explicit
`chat_template.model` marker backed by a model-specific C++ renderer. This is
used for contracts such as Qwen3-TTS and Alpamayo. Provider Jinja always takes
precedence when a checkpoint supplies it, including the current Qwen3-ASR and
Qwen3-Omni processor checkpoints.

## Runtime Context

The C++ renderer receives the structured request directly. Templates can use:

- `messages`, normalized to the provider template's string or content-block contract
- `message.reasoning` and the legacy alias `message.reasoning_content`
- `message.tool_calls`, tool-call IDs, names, and arguments
- `tools`, `tool_choice`, and `parallel_tool_calls`
- `add_generation_prompt`, `enable_thinking`, and `reasoning_effort`
- `bos_token` and `eos_token`

Image, video, audio, and trajectory blocks remain structured until the provider
template emits that model's placeholder tokens, except where an explicit raw
processor contract requires pre-render conversion. As in vLLM, JSON-encoded
assistant tool arguments are parsed into provider-visible objects and empty
`tool_calls` arrays use the ordinary assistant-message path. A `developer`
message is preserved when the provider template supports that role; otherwise,
it is converted to `system` and system messages are consolidated as in vLLM.

Template changes must be checked against the provider's
`apply_chat_template(..., tokenize=False)` output for text, system prompts,
multimodal content, assistant history, reasoning, tool calls, tool responses,
and generation-prompt modes supported by the family.
