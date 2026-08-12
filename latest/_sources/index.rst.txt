.. TensorRT Edge-LLM documentation master file, created by
   sphinx-quickstart on Wed Oct  8 16:38:08 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TensorRT Edge-LLM Documentation
================================

TensorRT Edge-LLM provides optimized inference for text, vision, audio, speech,
and action models on NVIDIA edge platforms.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   overview.md
   user_guide/getting_started/support-matrix.md
   user_guide/getting_started/supported-models.md
   user_guide/getting_started/installation.md
   user_guide/getting_started/quick-start-guide.md
   user_guide/getting_started/direct-engine-builder.md
   user_guide/getting_started/limitations.md

.. toctree::
   :maxdepth: 2
   :caption: Examples

   user_guide/examples/index.md
   user_guide/examples/speculative-decoding.md
   user_guide/examples/phi4.md
   user_guide/examples/asr.md
   user_guide/examples/tts.md
   user_guide/examples/vla/index.md
   user_guide/examples/omni.md
   user_guide/examples/experimental-server.md
   user_guide/examples/nemo-evaluator.md

.. toctree::
   :maxdepth: 2
   :caption: Features

   user_guide/features/lora.md
   user_guide/features/quantization.md
   user_guide/features/reduce-vocab.md
   user_guide/features/FP8KV.md
   user_guide/features/fp8-embedding.md
   user_guide/features/streaming.md
   user_guide/features/kv-cache-reuse.md

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
   developer_guide/software-design/onnxless-builder.md
   developer_guide/software-design/quantization-design.md
   developer_guide/software-design/engine-builder.md
   developer_guide/software-design/cpp-runtime-overview.md
   developer_guide/software-design/llm-inference-runtime.md
   developer_guide/software-design/llm-streaming.md

.. toctree::
   :maxdepth: 2
   :caption: Models

   developer_guide/models/cosmos3.md
   developer_guide/models/nemotron3_5_asr.md

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
