.. TensorRT Edge-LLM documentation master file, created by
   sphinx-quickstart on Wed Oct  8 16:38:08 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TensorRT Edge-LLM Documentation
================================

Welcome to the TensorRT Edge-LLM documentation. This library provides optimized inference capabilities 
for large language models and vision-language models on edge devices.

Getting Started
---------------

Get up and running with TensorRT Edge-LLM. Learn about platform overview, key features, use cases, 
supported platforms, and complete installation instructions for Python and C++ components.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   developer_guide/01.1_Overview.md
   developer_guide/01.2_Quick_Start_Guide.md
   developer_guide/01.3_Installation.md

Models
------

Learn about supported model families and architectures.

.. toctree::
   :maxdepth: 2
   :caption: Models

   developer_guide/02_Supported_Models.md

Model Export & Engine Building
-------------------------------

Convert and optimize your models for deployment. Learn how to convert HuggingFace models to ONNX 
with quantization and compile them into optimized TensorRT engines.

.. toctree::
   :maxdepth: 2
   :caption: Model Export & Engine Building

   developer_guide/03.1_Python_Export_Pipeline.md
   developer_guide/03.2_Engine_Builder.md

Chat Template Configuration
----------------------------

Learn how to create and customize chat templates for formatting conversational messages for your models.

.. toctree::
   :maxdepth: 2
   :caption: Chat Template Configuration

   developer_guide/06_Chat_Template_Format.md

C++ Runtime
-----------

Explore the C++ inference runtime and its capabilities, including runtime architecture, standard runtime 
for text and multimodal inference, EAGLE speculative decoding, CUDA graphs, LoRA, and batch processing.

.. toctree::
   :maxdepth: 2
   :caption: C++ Runtime

   developer_guide/04.1_C++_Runtime_Overview.md
   developer_guide/04.2_LLM_Inference_Runtime.md
   developer_guide/04.3_LLM_Inference_SpecDecode_Runtime.md
   developer_guide/04.4_Advanced_Runtime_Features.md

Examples
--------

Reference implementations demonstrating LLM, multimodal, and utility use cases.

.. toctree::
   :maxdepth: 2
   :caption: Examples

   developer_guide/05_Examples.md

Customization
-------------

Learn how to customize and extend TensorRT Edge-LLM for your specific needs.

.. toctree::
   :maxdepth: 2
   :caption: Customization

   developer_guide/07_Customization_Guide.md
   
TensorRT Plugins
----------------

Learn about the usage of TensorRT plugins with TensorRT Edge-LLM and how to make further customizations.

.. toctree::
   :maxdepth: 2
   :caption: TensorRT Plugins

   developer_guide/08_TensorRT_Plugins.md


APIs
----

API documentation for Python and C++ components.

.. toctree::
   :maxdepth: 2
   :caption: APIs

   python_api
   cpp_api

----

**Need help?** Visit our `GitHub repository <https://github.com/NVIDIA/TensorRT-Edge-LLM>`_ for issues and discussions.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
