<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Examples

New to Edge-LLM? Follow [Quick Start](../getting_started/quick-start-guide.md)
to start a chat or launch an HTTP server, then explore the examples below.

Choose the workflow by its input and output contract. The
[Supported Models](../getting_started/supported-models.md) page maps each model
family to these contracts.

Use the [trtedge CLI](../getting_started/cli.md) for chat,
serving, engine building, ONNX export, and model preparation.

## Text Output

- [Interactive Text Chat](interactive-chat.md): local streaming conversations with `trtedge chat`
- [Text generation](../getting_started/quick-start-guide.md): one shared workflow
  for dense LLM, MoE, VLM, and Omni text output; Phi-4 Multimodal adds the
  model-specific [vision-LoRA merge](phi4.md)
- [Python runtime](python-runtime.md): build a text engine and run inference from local model files
- [Speculative Decoding](speculative-decoding.md): MTP, EAGLE3, DFlash, DSpark,
  and JetSpec acceleration
- [ASR](asr.md): audio to transcript text with Qwen3-ASR
- [Nemotron-3.5-ASR](../../developer_guide/models/nemotron3_5_asr.md): audio to transcript text with a model-specific RNN-T runtime

## Speech Output

- [Qwen3-TTS](tts.md): text, style, language, or reference speech to 24 kHz speech
- [Qwen3-Omni](omni.md): text, images, or audio to text and optional 24 kHz speech

## Video Output

- [MiniMax-H3](minimax-h3.md): text to video with stereo audio from the FP16 checkpoint

## Action Output

- [Vision-Language-Action](vla/index.md): Alpamayo trajectory generation, and
  Cosmos3 and pi0.5 policy action generation

## Use with other software

- [Integration](integration/index.md): AIPerf, NeMo Evaluator, VLMEvalKit, Claude Code, OpenClaw, and Codex
- [Python API and Server](server.md): Python and OpenAI-compatible interfaces
