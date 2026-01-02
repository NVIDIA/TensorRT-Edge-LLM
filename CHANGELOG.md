# Release Notes
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
