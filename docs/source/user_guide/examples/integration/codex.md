<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Use Codex

[Codex CLI](https://developers.openai.com/codex) speaks the OpenAI Responses
API. Current releases accept only `wire_api = "responses"`. The server
translates `POST /v1/responses` onto chat completions, including client
function tools. `input_image` blocks become chat `image_url` parts, including
images returned by a tool. Streamed turns send each token as it is generated.
The server does not run server-side tools such as web search.

Use the [recommended sequence length](index.md). Codex sends its instructions,
skills, and tool list on every turn, about 14,000 tokens before the reply, and
that profile leaves room for the prompt and a short completion. Change either
flag if your prompt is larger or you want a smaller engine.

```bash
edgellm serve Qwen/Qwen3.5-4B \
  --max-input-len 32768 \
  --max-kv-cache-capacity 40960 \
  --enable-auto-tool-choice \
  --served-model-name qwen35-local \
  --port 8000
```

Point Codex at the server from `~/.codex/config.toml`. Provider settings belong
in that user config, not in a project `.codex/config.toml`.

```toml
model = "qwen35-local"
model_provider = "edgellm"

[model_providers.edgellm]
name = "TensorRT Edge-LLM"
base_url = "http://127.0.0.1:8000/v1"
env_key = "EDGE_LLM_API_KEY"
wire_api = "responses"
```

```bash
export EDGE_LLM_API_KEY=local
codex exec "Reply with the single word ready."
```

`model` must match `--served-model-name`. Set `EDGE_LLM_API_KEY` to any
non-empty value when the server omits `--api-key`, and to the same value when
the server sets `--api-key`. `--enable-auto-tool-choice` is required because
Codex sends tool definitions.
