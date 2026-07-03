# Limitations and Known Issues

This page documents the known limitations and issues for each release version of TensorRT Edge-LLM.

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
- Fixed in 0.7.0: Setting `--maxBatchSize=1` could cause accuracy degradation or engine build failure for some models. Setting `--maxBatchSize=2` was the 0.6.0 workaround.
- Fixed in 0.7.0: CuTe DSL kernels could hang for some inputs when runtime batch size > 1, so the kernels were only enabled when runtime batch size = 1 in 0.6.0.
