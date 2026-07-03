# Overview

> **Repository:** [github.com/NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)

## What is TensorRT Edge-LLM?

TensorRT Edge-LLM is NVIDIA's high-performance C++ inference runtime for Large Language Models (LLMs) and Vision-Language Models (VLMs) on embedded platforms. It enables efficient deployment of state-of-the-art language models on resource-constrained devices such as NVIDIA Jetson, NVIDIA DRIVE, and NVIDIA DGX Spark platforms.

## Supported Platforms

### Hardware Platforms

**Officially Supported Platforms:**

| Platform | Software Release | Link |
|----------|------------------|------|
| NVIDIA Jetson Thor | JetPack 7.x | [JetPack Website](https://developer.nvidia.com/embedded/jetpack) |
| NVIDIA DRIVE Thor | NVIDIA DriveOS 7.2 | [NVIDIA DRIVE Developer](https://developer.nvidia.com/drive) |
| NVIDIA DGX Spark (GB10) | DGX Spark software stack | [NVIDIA DGX Spark Developer](https://developer.nvidia.com/topics/ai/dgx-spark) |
| NVIDIA Jetson Orin | JetPack 7.2 | [JetPack Website](https://developer.nvidia.com/embedded/jetpack) |

> **Note:** The platforms listed above are officially supported and tested. Jetson Orin supports FP16, INT8, and INT4 model precisions. For exact build flags by platform and JetPack release, see the [Installation Guide](user_guide/getting_started/installation.md).

**Compatible Platforms:**

| Platform | Software Release |
|----------|------------------|
| NVIDIA Jetson Orin | JetPack 6.2+ |

> **Note:** JetPack 7.2 is the supported Jetson Orin path. JetPack 6.2+ remains compatible for FP16, INT8, and INT4 workflows.


### Supported Model Families

TensorRT Edge-LLM supports the deployment of a wide selection of LLM/VLM/Omni/VLA checkpoints with speculative decoding draft support, including Qwen, Llama, InternVL, Phi, Gemma, Nemotron, Alpamayo, Cosmos, etc. For the complete support matrix, see **[Supported Models](user_guide/getting_started/supported-models.md)**.

---

## Key Features

- **🚀 High Performance**: Optimized CUDA kernels and TensorRT integration for minimum latency
- **💾 Memory Efficient**: Supporting 4-bit quantization for reduced memory footprint, with [FP8 KV cache](user_guide/features/FP8KV.md) support for additional memory savings
- **🔄 Production Ready**: C++-only runtime with no Python dependencies, designed for deployment on edge devices
- **🎯 Edge Optimized**: Built specifically for NVIDIA Jetson, DRIVE, and DGX Spark platforms with platform-specific optimizations
- **🔧 Rich Feature Set**: Supports [LoRA adapters](user_guide/features/lora.md), EAGLE3, MTP, and DFlash speculative decoding, [system prompt caching](user_guide/features/system-prompt-cache.md), vision-language models, and an [experimental high-level Python API/server](user_guide/examples/experimental-server.md)
- **📊 Complete Toolkit**: End-to-end workflow from checkpoint export to C++ runtime, with engine builder and examples

## Key Components

> **Code Location:** `tensorrt_edgellm/quantization/` (checkpoint quantization), `tensorrt_edgellm/` (ONNX export), `experimental/server/` (Python API/server), `cpp/` (runtime), `examples/` (C++ examples)

TensorRT Edge-LLM uses a three-stage pipeline:

```mermaid
%%{init: {'theme':'neutral', 'themeVariables': {'primaryColor':'#76B900','primaryTextColor':'#fff','primaryBorderColor':'#5a8f00','lineColor':'#666','edgeLabelBackground':'#ffffff','labelTextColor':'#000','clusterBkg':'#ffffff','clusterBorder':'#999'}}}%%

graph LR
    HF_MODEL[HuggingFace Models<br>*including pre-quantized<br>checkpoints*]
    PYTHON_EXPORT(Checkpoint-Based<br>Model Exporter)
    ONNX_MODEL[ONNX<br>Model]
    ENGINE_BUILDER(Engine Builder)
    TRT_ENGINE[TensorRT<br>Engines]
    CPP_RUNTIME(C++ Runtime)
    SAMPLES(Examples)
    APPLICATIONS(Applications)

    HF_MODEL --> PYTHON_EXPORT
    PYTHON_EXPORT --> ONNX_MODEL
    ONNX_MODEL --> ENGINE_BUILDER
    ENGINE_BUILDER --> TRT_ENGINE
    TRT_ENGINE --> CPP_RUNTIME
    CPP_RUNTIME --> SAMPLES
    SAMPLES --> APPLICATIONS

    classDef greyNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
    classDef nvNode fill:#76B900,stroke:#5a8f00,stroke-width:1px,color:#fff
    classDef darkNode fill:#ffffff,stroke:#999,stroke-width:1px,color:#333
    classDef inputNode fill:#f5f5f5,stroke:#999,stroke-width:1px,color:#333
    classDef itemNode fill:#ffffff,stroke:#999,stroke-width:1px,color:#333

    class HF_MODEL inputNode
    class ONNX_MODEL,TRT_ENGINE itemNode
    class PYTHON_EXPORT,ENGINE_BUILDER,CPP_RUNTIME nvNode
    class APPLICATIONS darkNode
    class SAMPLES nvNode
```

| Component | Description |
|-----------|-------------|
| **Quantization Package** | Creates quantized HuggingFace-style checkpoints for the checkpoint exporter. [Usage](user_guide/features/quantization.md), [Design](developer_guide/software-design/quantization-design.md) |
| **Checkpoint Exporter** | Reads HuggingFace checkpoints directly and exports ONNX artifacts. [Learn More](developer_guide/software-design/checkpoint-export.md) |
| **Experimental Python API and Server** | Provides a vLLM-style Python API and OpenAI-compatible server. [Learn More](user_guide/examples/experimental-server.md) |
| **Engine Builder** | C++-based application that compiles ONNX models into optimized TensorRT engines. [Learn More](developer_guide/software-design/engine-builder.md) |
| **C++ Runtime** | C++-based runtime that executes TensorRT engines with CUDA graphs, LoRA, and speculative decoding support. [Learn More](developer_guide/software-design/cpp-runtime-overview.md) |
| **Examples** | Reference implementations demonstrating LLM, multimodal, and utility use cases. See the [Quick Start Guide](user_guide/getting_started/quick-start-guide.md) and example guides in the User Guide. |

---

## Next Steps

Ready to get started with TensorRT Edge-LLM? Follow these steps:

1. **[Installation Guide](user_guide/getting_started/installation.md)** - Set up quantization and `tensorrt_edgellm` on your x86 host and build the C++ runtime on your edge device

2. **[Quick Start Guide](user_guide/getting_started/quick-start-guide.md)** - Run your first LLM inference in ~15 minutes with step-by-step instructions

3. **Examples** - Explore advanced workflows including [VLM inference](user_guide/examples/vlm.md), [speculative decoding](user_guide/examples/speculative-decoding.md), [ASR](user_guide/examples/asr.md), [MoE](user_guide/examples/moe.md), [TTS](user_guide/examples/tts.md), and [VLA model inference](user_guide/examples/vla.md)

---

**For questions or issues, visit our [TensorRT Edge-LLM GitHub repository](https://github.com/NVIDIA/TensorRT-Edge-LLM).**
