.. TensorRT Edge-LLM documentation master file, created by
   sphinx-quickstart on Wed Oct  8 16:38:08 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TensorRT Edge-LLM Documentation
================================

Welcome to the TensorRT Edge-LLM documentation. This library provides optimized inference capabilities
for large language models and vision-language models on edge devices.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   overview.md
   user_guide/getting_started/supported-models.md
   user_guide/getting_started/installation.md
   user_guide/getting_started/quick-start-guide.md
   user_guide/getting_started/limitations.md

.. toctree::
   :maxdepth: 2
   :caption: Examples

   user_guide/examples/index.md
   user_guide/examples/vlm.md
   user_guide/examples/speculative-decoding.md
   user_guide/examples/phi4.md
   user_guide/examples/asr.md
   user_guide/examples/moe.md
   user_guide/examples/tts.md
   user_guide/examples/vla.md
   user_guide/examples/experimental-server.md

.. toctree::
   :maxdepth: 2
   :caption: Features

   user_guide/features/lora.md
   user_guide/features/quantization.md
   user_guide/features/reduce-vocab.md
   user_guide/features/FP8KV.md
   user_guide/features/fp8-embedding.md
   user_guide/features/streaming.md
   user_guide/features/system-prompt-cache.md

.. toctree::
   :maxdepth: 2
   :caption: Input & Chat Format

   user_guide/format/input-format.md
   user_guide/format/chat-template-format.md

.. toctree::
   :maxdepth: 2
   :caption: Performance

   user_guide/performance/performance-benchmarks.md

.. toctree::
   :maxdepth: 2
   :caption: Software Design

   developer_guide/software-design/checkpoint-export.md
   developer_guide/software-design/quantization-design.md
   developer_guide/software-design/engine-builder.md
   developer_guide/software-design/cpp-runtime-overview.md
   developer_guide/software-design/llm-inference-runtime.md
   developer_guide/software-design/llm-streaming.md

.. toctree::
   :maxdepth: 2
   :caption: Customization

   developer_guide/customization/customization-guide.md
   developer_guide/customization/calibration-datasets.md
   developer_guide/customization/tensorrt-plugins.md

.. toctree::
   :maxdepth: 2
   :caption: Testing

   developer_guide/testing/code-coverage.md
   developer_guide/testing/few-layer-validation.md

.. toctree::
   :maxdepth: 2
   :caption: APIs

   python_api
   cpp_api

.. toctree::
   :maxdepth: 2
   :caption: Quick Links

   Releases <https://github.com/NVIDIA/TensorRT-Edge-LLM/releases>
   GitHub <https://github.com/NVIDIA/TensorRT-Edge-LLM>
   Roadmap <https://github.com/NVIDIA/TensorRT-Edge-LLM/issues?q=is%3Aissue%20state%3Aopen%20label%3ARoadmap>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
