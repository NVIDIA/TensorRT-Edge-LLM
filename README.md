<div align="center">

# TensorRT Edge-LLM

**High-Performance Large Language Model Inference Framework for NVIDIA Edge Platforms**

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://nvidia.github.io/TensorRT-Edge-LLM/)
[![version](https://img.shields.io/badge/release-0.10.1-green)](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/tensorrt_edgellm/_version.py)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/LICENSE)

[Overview](https://nvidia.github.io/TensorRT-Edge-LLM/latest/overview.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Support Matrix](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/support-matrix.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Quick Start](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/quick-start-guide.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Performance](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/performance/performance-benchmarks.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](https://nvidia.github.io/TensorRT-Edge-LLM/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Roadmap](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues?q=is%3Aissue%20state%3Aopen%20label%3ARoadmap)

---
<div align="left">

## Latest News

- **[2026/09]** Release **0.10.1** adds experimental [**TP=2 inference on Dual NVIDIA DGX Spark**](docs/source/user_guide/features/multi-device.md) and redesigns the experimental [OpenAI-compatible server](docs/source/user_guide/examples/experimental-server.md) for faster cold launches and lower memory usage.
- **[2026/08]** TensorRT Edge-LLM **0.10.0** adds Day-0 support for [**Qwen3.8-27B**](https://huggingface.co/Qwen/Qwen3.8-27B).
- **[2026/08]** Release **0.10.0** adds support for [**NVIDIA Nemotron-3.5 Lightning**](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) with **MTP** and **DFlash**, [**Cosmos3-Edge**](https://huggingface.co/nvidia/Cosmos3-Edge), [**DiffusionGemma**](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4), [**Nemotron-3.5-ASR**](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b), and [**DSpark**](docs/source/user_guide/examples/speculative-decoding.md#dspark) speculative decoding, alongside an experimental [direct TensorRT engine builder](docs/source/user_guide/getting_started/direct-engine-builder.md) without ONNX export, multi-turn KV-cache reuse, and video input for the experimental OpenAI-compatible server.
- **[2026/07]** Support for the full **Gemma 4** family (E2B / E4B / 12B / 26B-A4B / 31B — multimodal text + image + audio, with MTP), **Qwen3-Omni** and **Nemotron-3** NVFP4, and **DFlash** speculative decoding (with DDTree for Qwen3 / Qwen3.5) landed across releases 0.9.0 and 0.9.1.

---

## Overview

TensorRT Edge-LLM is NVIDIA's C++ inference runtime for text, vision, audio, speech, and action models on NVIDIA Jetson, NVIDIA DRIVE, and NVIDIA DGX Spark. The supported frontend exports Hugging Face checkpoints to [ONNX](https://onnx.ai) for C++ engine building; an experimental direct frontend builds engines from checkpoints without ONNX. Both paths use the same C++ deployment runtimes.

---

## Getting Started

Check the [**Official Support Matrix**](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/support-matrix.html), then follow the [**Quick Start Guide**](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/quick-start-guide.html). Checkpoint IDs are listed in [**Supported Models**](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/supported-models.html).

---

## Documentation

### Introduction

- **[Overview](https://nvidia.github.io/TensorRT-Edge-LLM/latest/overview.html)** - What is TensorRT Edge-LLM and key features
- **[Official Support Matrix](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/support-matrix.html)** - Platform, JetPack, DriveOS, CUDA, TensorRT, and TensorRT Edge-LLM compatibility
- **[Supported Models](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/supported-models.html)** - Complete model compatibility matrix
- **[Checkpoint Exporter](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/software-design/checkpoint-export.html)** - Recommended ONNX export pipeline
- **[Experimental Direct Engine Builder](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/direct-engine-builder.html)** - Build all model components directly from a checkpoint

### User Guide

- **[Installation](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/installation.html)** - Set up quantization, `tensorrt_edgellm`, and the C++ runtime
- **[Quick Start Guide](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/quick-start-guide.html)** - Run your first inference in ~15 minutes
- **[Examples](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/examples/index.html)** - End-to-end workflows
- **[Quantization](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/features/quantization.html)** - Create quantized checkpoints for `tensorrt_edgellm`
- **[Experimental High-Level Python API and Server](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/examples/experimental-server.html)** - vLLM-style API and OpenAI-compatible server
- **[Input Format Guide](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/format/input-format.html)** - Request format and specifications
- **[Chat Template Format](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/format/chat-template-format.html)** - Chat template configuration

### Developer Guide

#### Software Design

- **[Quantization Package Design](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/software-design/quantization-design.html)** - Quantization package architecture
- **[Engine Builder](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/software-design/engine-builder.html)** - Building TensorRT engines
- **[C++ Runtime Overview](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/software-design/cpp-runtime-overview.html)** - Runtime system architecture
  - [LLM Inference Runtime](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/software-design/llm-inference-runtime.html)

#### Advanced Topics

- **[Customization Guide](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/customization/customization-guide.html)** - Customizing TensorRT Edge-LLM for your needs
- **[TensorRT Plugins](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/customization/tensorrt-plugins.html)** - Custom plugin development
- **[Tests](tests/)** - Comprehensive test suite for contributors

---

## Performance

See the [**Performance Benchmarks**](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/performance/performance-benchmarks.html) page for released benchmark results covering LLM and VLM prefill, generation throughput, memory usage, and EAGLE speculative decoding speedups.

---

## Use Cases

**🚗 Automotive**
- In-vehicle AI assistants
- Voice-controlled interfaces
- Scene understanding
- Driver assistance systems

**🤖 Robotics**
- Natural language interaction
- Task planning and reasoning
- Visual question answering
- Human-robot collaboration

**🏭 Industrial IoT**
- Equipment monitoring with NLP
- Automated inspection
- Predictive maintenance
- Voice-controlled machinery

**📱 Edge Devices**
- On-device chatbots
- Offline language processing
- Privacy-preserving AI
- Low-latency inference

---

## Featured Websites

- [TensorRT Edge-LLM Jetson AI Lab tutorial](https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/)
- [Maximizing Memory Efficiency to Run Bigger Models on NVIDIA Jetson](https://developer.nvidia.com/blog/maximizing-memory-efficiency-to-run-bigger-models-on-nvidia-jetson/)
- [Build Next-Gen Physical AI with Edge-First LLMs for Autonomous Vehicles and Robotics](https://developer.nvidia.com/blog/build-next-gen-physical-ai-with-edge%E2%80%91first-llms-for-autonomous-vehicles-and-robotics/)
- [Accelerate AI Inference for Edge and Robotics with NVIDIA Jetson T4000 and NVIDIA JetPack 7.1](https://developer.nvidia.com/blog/accelerate-ai-inference-for-edge-and-robotics-with-nvidia-jetson-t4000-and-nvidia-jetpack-7-1/)
- [Accelerating LLM and VLM Inference for Automotive and Robotics with NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/)

Follow our [GitHub repository](https://github.com/NVIDIA/TensorRT-Edge-LLM) for the latest updates, releases, and announcements.

---

## Support

- **Documentation**: [Full Documentation](https://nvidia.github.io/TensorRT-Edge-LLM/)
- **Quick Start**: [Quick Start Guide](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/quick-start-guide.html)
- **Roadmap**: [Developer Roadmap](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues?q=is%3Aissue%20state%3Aopen%20label%3ARoadmap)
- **Issues**: [GitHub Issues](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues)
- **Discussions**: [GitHub Discussions](https://github.com/NVIDIA/TensorRT-Edge-LLM/discussions)
- **Forums**: [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)

---

## License

[Apache License 2.0](LICENSE)

---

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

---
