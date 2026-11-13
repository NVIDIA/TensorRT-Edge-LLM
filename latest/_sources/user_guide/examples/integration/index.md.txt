<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Integration

Each tool below has its own page. Omit `--cache-dir`: the model downloads into
the Hugging Face cache, and the engine is reused from the default Edge-LLM
cache.

```bash
edgellm serve Qwen/Qwen3.5-4B \
  --max-input-len 32768 \
  --max-kv-cache-capacity 40960 \
  --served-model-name qwen35-local \
  --port 8000
```

Generation clients use `--max-input-len 32768` and
`--max-kv-cache-capacity 40960`. The input limit covers the prompt, including
a client's system prompt and tools. The KV-cache capacity must also hold the
completion. Change either flag when a workload needs a different length.
AIPerf stays at 8,192 and 10,240: keep each trace inside that limit instead of
building a longer engine.

Add `--enable-auto-tool-choice` for Claude Code, OpenClaw, and Codex. The
server checks a bearer token only when `--api-key` is set.

| Tool | Endpoint |
|---|---|
| [AIPerf](aiperf.md) | `POST /v1/chat/completions` |
| [NeMo Evaluator](nemo-evaluator.md) | `POST /v1/chat/completions` |
| [VLMEvalKit](vlmevalkit.md) | `POST /v1/chat/completions` |
| [Claude Code](claude-code.md) | `POST /v1/messages` |
| [OpenClaw](openclaw.md) | `POST /v1/chat/completions` |
| [Codex](codex.md) | `POST /v1/responses` |

```{toctree}
:maxdepth: 1

aiperf.md
nemo-evaluator.md
vlmevalkit.md
claude-code.md
openclaw.md
codex.md
```
