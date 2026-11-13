# Limitations and Known Issues

This page documents the known limitations and issues for each release version of TensorRT Edge-LLM.

## 0.12.0

- On Windows with RTX GPUs (TensorRT-RTX 1.7), engine builds for some large NVFP4 MoE and dense models (27B-35B class) can fail with a TensorRT internal error in `safeMemcpyHostToDevice` (Error Code 2). The fix is planned for the next version of TensorRT-RTX.
- Windows on ARM is not supported. The Windows source build in [Installation](installation.md) targets x64 only.
- On Jetson Thor, prefill throughput for some models, including native, EAGLE, MTP, and DFlash execution with NVFP4 or FP8 weights, is lower than in 0.10.1. The 0.11 token-major MLP layout prevents TensorRT from fusing the gate and up projections into a dual GEMM. A fix is planned for a later release.
- NVFP4 quantization of models with a Conv3d vision tower, such as Qwen3-Omni-30B-A3B-Instruct and Alpamayo-R1-10B, fails during calibration with `Error building extension 'conv3d_implicit_gemm_cuda_v20_wmma'`. ModelOpt 0.45.0 compiles this extension as C++17, but the PyTorch 2.14.0 headers require C++20. Downgrade to PyTorch 2.13.0 in the quantization environment, with the matching torchvision. Other NVFP4 VLMs, and FP16 or INT4-AWQ checkpoints of these models, are not affected.
- The published NVFP4 (W4A16) Nemotron-3.5-Bolt-E4B DFlash draft has a low acceptance rate. Set `EDGELLM_DFLASH_DENSE_LAST_LAYER=1` for the ONNX export and for `trtedge build` to run the draft's last layer as dense FP16. The variable applies only to a Qwen3-style DFlash draft behind a Gemma4 target.

## 0.10.0

- Fixed in 0.10.1: `Int4GroupwiseGemmPluginV2` may cause accuracy degradation. For ONNX export, use `--int4-gemm-plugin-version 1` to select the V1 plugin as a fallback.

## 0.9.0

- Some FP8 ViT configurations may produce NaN outputs for specific models. If this occurs, use FP16 ViT for the visual encoder.

## 0.7.0

- Fixed in 0.7.1: `tensorrt-edgellm-quantize` exposed FP8 visual-tower quantization for supported VLM checkpoints, but other visual-tower precisions were not exposed in 0.7.0.
- Fixed in 0.7.1: Qwen3-TTS required a hybrid export workflow in 0.7.0.
- Reduced-vocabulary exports with packed INT4 LM heads require `group_size=128` and a reduced vocabulary size that is a multiple of 128.

## 0.6.1

- Some large models (7B-8B) may encounter cudaMallocAsync issue on DriveOS. Please increase huge page by `echo 15658 | sudo tee /proc/sys/vm/nr_hugepages`

## 0.6.0

- TensorRT 10.15 may cause accuracy degradation with NVFP4 for some models. Use TensorRT 10.13.3.9 shipped with Jetpack 7.1 instead.
- Fixed in 0.7.0: Setting `--max-batch-size=1` could cause accuracy degradation or engine build failure for some models. Setting `--max-batch-size=2` was the 0.6.0 workaround.
- Fixed in 0.7.0: CuTe DSL kernels could hang for some inputs when runtime batch size > 1, so the kernels were only enabled when runtime batch size = 1 in 0.6.0.
