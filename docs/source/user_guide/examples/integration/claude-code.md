<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Use Claude Code

`POST /v1/messages` and `POST /v1/messages/count_tokens` share the loaded model
and admission controller with OpenAI chat. The adapter supports system prompts,
text, base64 image blocks, client tools, `tool_use`/`tool_result`, thinking
output, and Anthropic SSE framing. Server-executed tools are not run by
TensorRT Edge-LLM.

Use the [recommended sequence length](index.md). Raise either limit if Claude
Code's prompt does not fit. Omit `--cache-dir` to use the default cache.
`--enable-auto-tool-choice` is required because Claude Code sends tool
definitions.

```bash
edgellm serve Qwen/Qwen3.5-4B \
  --max-input-len 32768 \
  --max-kv-cache-capacity 40960 \
  --enable-auto-tool-choice \
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

A one-shot `claude -p` reply on Qwen3.5-4B was `ready`. The matching
`POST /v1/messages` response was:

```json
{"content":[{"type":"text","text":"Ready."}],"stop_reason":"end_turn"}
```
