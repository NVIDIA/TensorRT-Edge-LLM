# Release Notes

## Unreleased
- Added Gemma 4 (`gemma4_text`) checkpoint recognition and architecture config parsing in the checkpoint export frontend; ONNX export and runtime support are in progress (see [#72](https://github.com/NVIDIA/TensorRT-Edge-LLM/issues/72))

## 0.8.0
- Externalized INT4 FFN, INT4 MoE, and LM-head weights to reduce engine build memory usage
- Upgraded plugins to TensorRT Plugin V3 for TensorRT 11 readiness
- Replaced the experimental exporter namespace with the checkpoint-based `tensorrt_edgellm` loader and export workflow
- Re-architected the C++ runtime with composable execution, decoding, preprocessing, and state-management components
- Added pluggable decoding strategies and per-request stop-string support
- Added tool-call parsing support for the experimental OpenAI-compatible server
- Added Qwen3.5/Qwen3.6 MoE INT4 and NVFP4 frontend support, including XQA head-dim 256 / KV-ratio 8 handling
- Updated `nvidia-cutlass-dsl` to 4.5.1
- Improved quantization robustness for multimodal calibration, ModelOpt quant-config lists, Qwen3.5 MTP export, and mixed-precision overrides
- Improved Qwen3-TTS, Qwen3-ASR, and LoRA export/test path validation
- Fixed XQA kernel loader thread safety and CUDA 13.3 MoE top-k softmax namespace build issues

## 0.7.1
- Added Qwen3.5 Multi-Token Prediction (MTP) support with performance improvements
- Added Alpamayo-1 support
- Added Qwen3-TTS streaming
- Added `llm_loader` FP8 ViT support
- Improved Mamba2 SSD prefill performance
- Added v0.7.0 performance benchmark results
- Migrated documentations and customization guides to `llm_loader`

## 0.7.0
- Added `llm_loader` reduced-vocabulary export support
- Added `llm_loader` static LoRA merge for merge -> quantize -> export workflows such as Phi-4-Multimodal
- Added Qwen3.5 LLM/VLM support
- Added Nemotron-Omni support
- Added Nemotron-3-Nano-30B-A3B NVFP4 MoE support
- Promoted the checkpoint-based `llm_loader` export path, including dynamic LoRA insertion and adapter weight processing
- Added a standalone quantization package for `llm_loader` checkpoints
- Added experimental high-level Python API and OpenAI-compatible server with streaming and EAGLE support
- Added FP8 embedding support to reduce embedding-table memory
- Reduced runtime memory by sharing TensorRT execution context memory
- Unified LLM runtime execution paths
- Added per-slot streaming with the `StreamChannel` API
- Improved build workflow with automatic TensorRT detection
- Upgraded Transformers support to 5.x

## 0.6.1
- Added DriveOS 7.2.4 official support
- Fixed EAGLE draft model weights loading issue to retrieve acceptance rate

## 0.6.0
- Added Nemotron-Nano-9B-v2 support via mamba_ssm and causal_conv1d
- Added Qwen3-30B-A3B-GPTQ-Int4 support via INT4 MoE Plugin
- Added Qwen3-ASR and Qwen3-TTS end-to-end support
- Added cutedsl FMHA kernels to speed up prefill performance on Blackwell
- Used ViT Attention Plugin with fmha-v2 and cutedsl kernels to speed up multi image ViT performance

## 0.5.0
- Implemented and used standalone embedding processing module to reduce multi-modal modeling complexity and reduce Eagle inference memory footprint
- Added FP8 KV Cache support
- Unified TensorRT execution context for prefill and decode to reduce memory footprint
- Supported vanilla decoding for speculative decoding runtime
- Used collision resistant hashing for CUDA graphs
- Updated int4GroupwiseGemmPlugin to TensorRT Plugin-v3 interface
- Refactored documentations
- Added ViT attention mask and RoPE parameter caching to reduce recomputation for Qwen
- Added Jetpack 6.2 compatibility 

## 0.4.0
- Refactored AttentionPlugin to use Tensor class and clearer shape checks
- Added support for multi-batch EAGLE3
- Enabled Open-AI style chat template
- Enabled vocab reduction to improve lm_head time
- Removed deprecated plugin fields
- Added Qwen3-VL support
- Added Phi-4-Multimodal support

## 0.3.0
- Refactored the vanilla decoding and EAGLE3 runtime to use consumer-producer design with `Tensor` class to manage all runtime memory
- Refactored and added unit tests for eagle utility and sampling kernels
- Refactored the example to use `llm_inference` as the only entry point for running inference
- Used json format for all the inputs and outputs for `llm_inference`
- Refactored the benchmark and include the performance metrics in `llm_inference` 
- Refactored engine builder and moved `builder` module into `cpp` folder
- Refactored the Python package to use `tensorrt-edgellm-quantize-llm`, `tensorrt-edgellm-export-llm`, etc. to export the model instead of native script and add `pip` support for Python
- Implemented torch custom op for AttentionPlugin instead of using onnx_graphsurgeon
- Bumped `nvidia-modelopt` and `transformers` package for various bug fixes
- Improved EAGLE3 acceptance rate and performance
- Added Qwen3 dense model support
- Added nvfp4&fp8 lm_head quantization, int4_gptq checkpoint support
- Added formal accuracy benchmark processes in `examples/accuracy` folder
- Added and refactored `tests` folder to run end-to-end pipeline tests
- Added `kernelSrcs` folder with `fmha` and `xqa` kernel cubin generation logic
- Improved all documentations and added developer's guide
- Added doxygen generated API docs and unified doc into `docs` folder
- Added safetensorUtils to read and write to safeTensor for debugging, LoRA weights and d2t weights loading
- Removed merged and static LoRA support
- Removed EAGLE2 support
- Removed all legacy code
- Removed dummy value benchmark script

## 0.2.0
- Added formal CUDA13.0 support
- Refactored SM120 and SM121 support
- Unified `int32_t` for input_ids and `float` for logits
- Replaced `jsmn` with `nlohmann/json` for better Json read/write support.
- Improved Attention performance by passing `rope_rotary_cos_sin` as model inputs
- Supported longrope
- Refactored Multimodal Runners and added them into `cpp` folder
- Improved runtime parsing from config files and folder structure
- Added runtime `Tensor` class

## 0.1.1
- Added EAGLE support for Qwen2.5-VL
- Added model support for DeepSeek-Distilled Qwen, InternVL3-1B
- Improved Sampler API
- Improved EAGLE pipeline

## 0.1.0
- Initial bring up of EAGLE2 & EAGLE3 with tree attention kernels
- Initial bring up of static and dynamic LoRA
- Added model support for Qwen2.5 3B, Qwen2.5-VL 3B, Qwen2.5-VL 7B
- Added FP8 VIT recipe for VLM
- Add JSON parser implementation
- Improved NVFP4 and FP8 performance with TensorRT10.10
- Improved unit tests and coding style
- Fixed C++ memory leak

## 0.0.3
- Fixed CUDA Graph capture errors
- Fixed NVFP4 accuracy issue by changing quantization recipe

## 0.0.2
- Added VLM support with Qwen2-VL-2B, Qwen2-VL-7B examples
- Added Qwen2-0.5B and Llama3-1B support by extending `AttentionPlugin`
- Added NVFP4 precision support for all models
- Added CUDA Graph support to improve inference latency for all models
- Improved INT4 performance by `Int4GroupwiseGemmPlugin`
- Improved usage and coding style
- Fixed C++ memory leak

## 0.0.1
- Completed end-to-end Llama and Qwen < 7B inference workflow with FP16, FP8 and INT4 support
    - Completed Python export script to quantize and export the LLM into desired precision and surgeon the graph to required format
    - Completed `AttentionPlugin` to support static shape attention with RoPE
    - Completed `decoder` and `sampler` to support end-to-end LLM inference workflow with
    - Completed Llama and Qwen `tokenizer` support
    - Completed `examples` with `chat`, `benchmark` and `accuracy` to showcase the usage and benchmark accuracy and performance
