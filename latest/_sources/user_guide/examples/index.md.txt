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

End-to-end workflows demonstrating TensorRT Edge-LLM capabilities across different use cases.

## Available Examples

- **[VLM (Vision-Language Model)](vlm.md)** - Complete workflow for vision-language models with image understanding capabilities
- **[Speculative Decoding](speculative-decoding.md)** - EAGLE3, MTP, and DFlash speculative decoding for faster inference
- **[Phi-4 Multimodal](phi4.md)** - Phi-4-Multimodal deployment with LoRA merge
- **[ASR (Automatic Speech Recognition)](asr.md)** - Speech-to-text with Qwen3-ASR models, including optional FP8 / NVFP4 quantization recipes
- **[MoE (Mixture of Experts)](moe.md)** - Mixture of Experts model deployment
- **[TTS (Text-to-Speech)](tts.md)** - Text-to-speech synthesis workflows
- **[Alpamayo-R1-10B (VLA)](vla.md)** - Vision-language-action workflow with image, text, trajectory history, and action prediction
- **[Omni (Audio + Vision + Speech I/O)](omni.md)** - End-to-end Qwen3-Omni multimodal pipeline with NVFP4 quantization
- **[Experimental High-Level Python API and Server](experimental-server.md)** - vLLM-style API and OpenAI-compatible server with spec-decode support
