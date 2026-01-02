# TensorRT Edge-LLM

**High-Performance Large Language Model Inference Framework for NVIDIA Edge Platforms**

---

## Overview

TensorRT Edge-LLM is NVIDIA's high-performance C++ inference runtime for Large Language Models (LLMs) and Vision-Language Models (VLMs) on embedded platforms. It enables efficient deployment of state-of-the-art language models on resource-constrained devices such as NVIDIA Jetson and NVIDIA DRIVE platforms. TensorRT Edge-LLM provides convenient Python scripts to convert HuggingFace checkpoints to [ONNX](https://onnx.ai). Engine build and end-to-end inference runs entirely on Edge platforms.

---

## Getting Started

For the supported platforms, models and precisions, see the [**Overview**](docs/source/developer_guide/01.1_Overview.md). Get started with TensorRT Edge-LLM in <15 minutes. For complete installation and usage instructions, see the [**Quick Start Guide**](docs/source/developer_guide/01.2_Quick_Start_Guide.md).

---

## Documentation

### Developer Guide

Complete documentation for installation, usage, and deployment:

- **[Overview](docs/source/developer_guide/01.1_Overview.md)** - What is TensorRT Edge-LLM and key features
- **[Quick Start Guide](docs/source/developer_guide/01.2_Quick_Start_Guide.md)** - Get started in ~15 minutes
- **[Installation](docs/source/developer_guide/01.3_Installation.md)** - Detailed installation instructions
- **[Supported Models](docs/source/developer_guide/02_Supported_Models.md)** - Complete model compatibility matrix
- **[Python Export Pipeline](docs/source/developer_guide/03.1_Python_Export_Pipeline.md)** - Model export and quantization
- **[Engine Builder](docs/source/developer_guide/03.2_Engine_Builder.md)** - Building TensorRT engines
- **[C++ Runtime Overview](docs/source/developer_guide/04.1_C++_Runtime_Overview.md)** - Runtime system architecture
  - [LLM Inference Runtime](docs/source/developer_guide/04.2_LLM_Inference_Runtime.md)
  - [LLM SpecDecode Runtime](docs/source/developer_guide/04.3_LLM_Inference_SpecDecode_Runtime.md)
  - [Advanced Runtime Features](docs/source/developer_guide/04.4_Advanced_Runtime_Features.md)
- **[Examples](docs/source/developer_guide/05_Examples.md)** - Working code examples
- **[Chat Template Format](docs/source/developer_guide/06_Chat_Template_Format.md)** - Chat template configuration
- **[TensorRT Plugins](docs/source/developer_guide/07_TensorRT_Plugins.md)** - Introduction for TensorRT plugins.


### Additional Resources

- **[Examples Directory](examples/)** - LLM and VLM inference examples
- **[Tests](tests/)** - Comprehensive test suite for contributors

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

## Tech Blogs

*Coming soon*

Stay tuned for technical deep-dives, optimization guides, and deployment best practices.

---

## Latest News

*Coming soon*

Follow our [GitHub repository](https://github.com/NVIDIA/TensorRT-Edge-LLM) for the latest updates, releases, and announcements.

---

## Support

- **Documentation**: [Developer Guide](docs/source/developer_guide/01.1_Overview.md)
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
