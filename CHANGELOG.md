# Release Notes

## 0.10.1
- Added experimental Dual NVIDIA DGX Spark support with TP=2 tensor-parallel inference over MPI and NCCL
- Redesigned the experimental OpenAI-compatible server for faster cold launches and lower memory usage
- Added architecture-aware Python wheel packaging for x86, Jetson, DRIVE, and DGX Spark
- Added Nemotron-Omni video input to the experimental server
- Added JetSpec speculative decoding support for Qwen3-8B
- Added Qwen3.8-27B DSpark hybrid-base export support
- Extended MTP verification trees to depth 16 and added `logit_bias` support on speculative decoding paths
- Added context-cache reuse for speculative decoding, including hybrid and MTP models
- Added end-to-end FP8 ViT attention support
- Added DART visual-token pruning to reduce VLM prefill work
- Added a media artifact cache that retains ViT and audio encoder embeddings across repeated media prefixes
- Added NVFP4 W4A16 export for official LM heads and Qwen3 A16 MoE
- Added NVFP4 MoE build and runtime support for NVIDIA Nemotron 3 Super 120B-A12B checkpoints with 512 routed experts and top-32 routing
- Added Blackwell SM110 dense NVFP4-A16 GEMM and GEMV kernels
- Improved Blackwell performance with single-launch D80 SSD prefill, faster SM12x fused MoE, fused GDN projections, and one less per-layer KV copy
- Added a CuTe DSL RMSNorm plugin and upgraded CuTe DSL to 4.7.0
- Enabled FMHA-v2 and XQA on Hopper SM90 with JIT-compiled XQA kernels
- Added QNX cross-compilation support and DGX Spark platform detection
- Fixed `Int4GroupwiseGemmPluginV2` accuracy for QKV concat aliasing, split-K workspace isolation, and LoRA insertion
- Fixed speculative decoding page tables, EAGLE KV page planes, DSpark paged KV metadata, Mamba replay, and draft cost accounting
- Fixed context reuse for cross-batch image contamination and zero-budget encoder caches
- Fixed export for Gemma 4 Unified quantized checkpoints, Qwen NVFP4 mixed-precision dispatch, fused-MoE MTP drafts, and thinking-default chat templates
- Fixed the Cosmos3 Reasoner visual patch layout and batched audio embedding concatenation
- Fixed cross-platform GPU memory reporting

## 0.10.0
- Added Day-0 support for Qwen3.8-27B
- Added NVIDIA Nemotron-3.5 Lightning support with MTP and DFlash speculative decoding
- Added an experimental ONNX-less workflow to build TensorRT engines directly from supported checkpoints
- Added Cosmos3-Edge multimodal reasoning and experimental policy/action-generation support
- Added DiffusionGemma block-diffusion inference support
- Added Nemotron-3.5-ASR support
- Added DSpark speculative decoding with DDTree drafting
- Added paged KV-cache reuse for multi-turn LLM requests and media-aware KV reuse for VLM requests
- Added video input support to the experimental server
- Expanded Qwen3-TTS with language and instruction conditioning, VoiceDesign, and on-device voice cloning
- Replaced the legacy embedded-cubin FMHA-v2 backend with CuTe DSL FMHA-v2 and removed the checked-in FMHA-v2 cubin artifacts
- Added paired-checkpoint MTP support for Gemma 4 12B
- Added Gemma 4 26B-A4B INT4 AWQ MoE quantization and export support
- Added Qwen3-Omni 30B-A3B INT4 AWQ quantization and export support
- Added Qwen3-Omni streaming audio output, `/v1/audio/speech` for Qwen3-TTS, and `/v1/audio/transcriptions` for Qwen3-ASR
- Added Claude Code and OpenClaw integration to the experimental server
- Fixed DFlash verification
- Fixed MTP/GDN accepted-state replay

## 0.9.1
- Added Gemma 4 12B, 26B-A4B, and 31B full support with MTP
- Expanded DFlash with DDTree for Qwen3 and Qwen3.5 hybrid models, plus 128K/64K/32K draft vocabulary reduction
- Added paged KV-cache prefill and paged XQA decode support
- Added Qwen video preprocessing and online GPU filter-bank preprocessing for Whisper-based Qwen3 audio models and Parakeet-based Nemotron-Omni
- Added logprobs across vanilla and speculative decoders and OpenAI-compatible `logit_bias` requests
- Added INT4-FP16 CuTe DSL GEMM/GEMV kernels, FP16 MoE support for 128/256 experts, and head-dimension 256 FMHA
- Added named, per-modality calibration datasets
- Added Qwen3-Omni/Qwen3-TTS CodePredictor FP8 quantization
- Fixed speculative-decoding sampling and engine routing, Gemma 4 export issues, FP8 long-context accuracy, and runtime synchronization/stability bugs

## 0.9.0
- Added Gemma 4 E2B/E4B text inference support
- Added Gemma4 Unified 12B image and audio multimodal inference support
- Added DFlash support
- Added Qwen3-Omni-30B-A3B NVFP4 support
- Fixed decode performance regression in 0.8.0
- Added Nemotron3 NVFP4 support for Jetson Thor
- Added C++ audio decoding/mel preprocessing and server audio input for supported audio models
- Expanded NVFP4 MoE backend coverage for SM100/101/110
- Added XQA sliding-window support and head-dim 512 kernels across supported SM targets

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
