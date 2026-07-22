<div align="center">

# TensorRT Edge-LLM

**High-Performance Large Language Model Inference Framework for NVIDIA Edge Platforms**

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://nvidia.github.io/TensorRT-Edge-LLM/)
[![version](https://img.shields.io/badge/release-0.9.0-green)](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/tensorrt_edgellm/_version.py)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/LICENSE)

[Overview](https://nvidia.github.io/TensorRT-Edge-LLM/latest/overview.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Quick Start](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/quick-start-guide.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Performance](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/performance/performance-benchmarks.html)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](https://nvidia.github.io/TensorRT-Edge-LLM/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Roadmap](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues?q=is%3Aissue%20state%3Aopen%20label%3ARoadmap)

---
<div align="left">

## Overview

TensorRT Edge-LLM is NVIDIA's high-performance C++ inference runtime for Large Language Models (LLMs), Vision-Language Models (VLMs), Vision-Language-Action (VLA) models, and World Foundation Models (WFMs) on embedded platforms. It enables efficient deployment of state-of-the-art language models on resource-constrained devices such as NVIDIA Jetson, NVIDIA DRIVE, and NVIDIA DGX Spark platforms. TensorRT Edge-LLM provides convenient Python scripts to convert HuggingFace checkpoints to [ONNX](https://onnx.ai). Engine build and end-to-end inference runs entirely on Edge platforms.

---

## Getting Started

For the supported platforms, models and precisions, see the [**Overview**](https://nvidia.github.io/TensorRT-Edge-LLM/latest/overview.html). Get started with TensorRT Edge-LLM in <15 minutes. For complete installation and usage instructions, see the [**Quick Start Guide**](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/quick-start-guide.html).

---

## World Foundation Models (Cosmos3-Edge)

TensorRT Edge-LLM includes `WFMInferenceRuntime` and the `wfm_inference` example for text-to-video generation with [nvidia/Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge). The runtime chains five TRT engines (VAE encode/decode, vision embed, MoT backbone, denoise head) behind a single C++ API.

### Build `wfm_inference`

After building the C++ runtime and Edge-LLM TensorRT plugins (see [Installation](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/installation.html)):

```bash
cd /path/to/TensorRT-Edge-LLM/build
cmake --build . --target wfm_inference -j$(nproc)
```

### Export engines

Cosmos3-Edge engines are exported with the companion script `Test/wfm/export_wfm_cosmos_edge.py` (Torch-TensorRT; requires a CUDA GPU with enough memory for the ~3.9B MoT backbone). From the `Test` export workspace:

```bash
cd /path/to/Test
export EDGE_LLM_PLUGIN_SO=/path/to/TensorRT-Edge-LLM/build-plugin-trt11/libNvInfer_edgellm_plugin.so

python wfm/export_wfm_cosmos_edge.py \
  --engine-dir /tmp/cosmos_edge_engines \
  --dtype fp16 \
  --num-inference-steps 2 \
  --prompt "A robot arm picks up a red cube."
```

This writes an engine bundle:

```
<engine_dir>/
  config.json
  packing_static.json
  embedding.safetensors
  tokenizer/
  visual_encode/visual_encode.engine
  embed/embed.engine
  mot_backbone/mot_backbone.engine
  denoise_head/denoise_head.engine
  visual_decode/visual_decode.engine
```

Use the same `--prompt` at export and inference time so `packing_static.json` matches the runtime text packing (`und_len` / sequence length).

### Run inference

```bash
export EDGELLM_PLUGIN_PATH=/path/to/TensorRT-Edge-LLM/build-plugin-trt11/libNvInfer_edgellm_plugin.so

./build/examples/wfm/wfm_inference \
  --engineDir=/tmp/cosmos_edge_engines \
  --inputFile=examples/wfm/wfm_input_example.json \
  --outputFile=/tmp/wfm_output.json
```

See `examples/wfm/wfm_input_example.json` for the request JSON format. Each request specifies a text `prompt`, optional `output_video_file` (raw fp16 tensor dump), `num_inference_steps`, and `seed`. Set `generate_sound: true` only for Cosmos3-Omni bundles that include audio engines.

---

## Documentation

### Introduction

- **[Overview](https://nvidia.github.io/TensorRT-Edge-LLM/latest/overview.html)** - What is TensorRT Edge-LLM and key features
- **[Supported Models](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/supported-models.html)** - Complete model compatibility matrix
- **[Checkpoint Exporter](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/software-design/checkpoint-export.html)** - Recommended ONNX export pipeline

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
- World-model video generation (Cosmos3-Edge WFM)

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
