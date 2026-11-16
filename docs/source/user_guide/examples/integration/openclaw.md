<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Use OpenClaw

[OpenClaw](https://docs.openclaw.ai/gateway/config-tools/custom-providers)
uses an OpenAI-compatible chat endpoint. Use the
[recommended sequence length](index.md), and raise either limit if OpenClaw's
prompt does not fit. Serve a model id without a slash, and enable automatic
tool choice because OpenClaw sends tools.

```bash
edgellm serve Qwen/Qwen3.5-4B \
  --max-input-len 32768 \
  --max-kv-cache-capacity 40960 \
  --enable-auto-tool-choice \
  --served-model-name qwen35-local \
  --port 8000
```

Add this provider to `~/.openclaw/openclaw.json`:

```json5
{
  models: {
    providers: {
      edgellm: {
        baseUrl: "http://127.0.0.1:8000/v1",
        api: "openai-completions",
        apiKey: "local",
        models: [
          {
            id: "qwen35-local",
            name: "Qwen3.5 4B",
            input: ["text"],
            contextWindow: 40960,
            maxTokens: 2048,
          },
        ],
      },
    },
  },
  agents: {
    defaults: {
      model: { primary: "edgellm/qwen35-local" },
    },
  },
}
```

`baseUrl` must end in `/v1`. The primary model id is `edgellm/qwen35-local`.
That origin is trusted for this provider. Match `apiKey` to `--api-key` when
the server requires one. The server ignores the bearer token when `--api-key`
is omitted.
