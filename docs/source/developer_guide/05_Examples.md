# Examples

**Previous**: [Advanced Runtime Features](04.4_Advanced_Runtime_Features.md)

---

> **Code Location:** `examples/` | **Build:** `examples/llm/`, `examples/multimodal/`

## Table of Contents

- [Overview](#overview)
- [Example Flow](#example-flow)
- [Build Examples](#build-examples)
- [Inference Examples](#inference-examples)
- [Complete Workflows](#complete-workflows)
- [Input File Formats](#input-file-formats)
- [Common Parameters](#common-parameters)

---

## Overview

C++ examples for building engines and running inference. All examples include detailed README files and source code in `examples/`.

> **⚠️ USER RESPONSIBILITY**: Users are responsible for composing meaningful and appropriate prompts for their use cases. The examples provided demonstrate technical usage patterns but do not guarantee output quality or appropriateness.

---

## Example Flow

```
ONNX Models → [Build Examples] → TensorRT Engines → [Inference Examples] → Results
```

---

## Build Examples

### `llm_build` - Source: `examples/llm/llm_build.cpp`

Builds TensorRT engines for LLMs (standard, EAGLE, VLM, LoRA).

```bash
# Standard LLM
./build/examples/llm/llm_build \
  --onnxDir onnx_models/qwen3-4b \
  --engineDir engines/qwen3-4b \
  --maxBatchSize 1

# Multimodal (VLM)
./build/examples/llm/llm_build \
  --onnxDir onnx_models/qwen2.5-vl-3b \
  --engineDir engines/qwen2.5-vl-3b \
  --maxBatchSize 1 \
  --maxInputLen=1024 \
  --maxKVCacheCapacity=4096 \
  --vlm \
  --minImageTokens 128 \
  --maxImageTokens 512

# EAGLE (speculative decoding)
./build/examples/llm/llm_build \
  --onnxDir onnx_models/model_eagle_base \
  --engineDir engines/model_eagle \
  --eagleBase

./build/examples/llm/llm_build \
  --onnxDir onnx_models/model_eagle_draft \
  --engineDir engines/model_eagle \
  --eagleDraft
```

### `visual_build` - Source: `examples/multimodal/visual_build.cpp`

Builds TensorRT engines for vision encoders (Qwen-VL, InternVL, Phi-4-Multimodal).

```bash
./build/examples/multimodal/visual_build \
  --onnxDir onnx_models/qwen2.5-vl-3b/visual_enc_onnx \
  --engineDir visual_engines/qwen2.5-vl-3b \
  --minImageTokens 128 \
  --maxImageTokens 512 \
  --maxImageTokensPerImage=512
```

---

## Inference Examples

### `llm_inference` - Source: `examples/llm/llm_inference.cpp`

Runs batch inference from JSON files. Supports standard, EAGLE, multimodal, and LoRA modes.

```bash
# Standard LLM
./build/examples/llm/llm_inference \
  --engineDir engines/qwen3-4b \
  --inputFile input.json \
  --outputFile output.json

# EAGLE (speculative decoding)
./build/examples/llm/llm_inference \
  --engineDir engines/model_eagle \
  --inputFile input.json \
  --outputFile output.json \
  --eagle

# Multimodal (VLM)
./build/examples/llm/llm_inference \
  --engineDir engines/qwen2.5-vl-3b \
  --multimodalEngineDir visual_engines/qwen2.5-vl-3b \
  --inputFile input_with_images.json \
  --outputFile output.json

# With profiling
./build/examples/llm/llm_inference \
  --engineDir engines/qwen3-4b \
  --inputFile input.json \
  --outputFile output.json \
  --dumpProfile \
  --profileOutputFile profile.json
```

---

## Complete Workflows

### Standard LLM (End-to-End)

```bash
# 1. Export ONNX (x86 host)
tensorrt-edgellm-quantize-llm --model_dir Qwen/Qwen3-4B-Instruct-2507 --quantization fp8 --output_dir quantized/qwen3-4b
tensorrt-edgellm-export-llm --model_dir quantized/qwen3-4b --output_dir onnx_models/qwen3-4b

# 2. Build Engine (Thor device)
./build/examples/llm/llm_build --onnxDir onnx_models/qwen3-4b --engineDir engines/qwen3-4b --maxBatchSize 1

# 3. Run Inference (Thor device)
./build/examples/llm/llm_inference --engineDir engines/qwen3-4b --inputFile input.json --outputFile output.json
```

### Multimodal VLM (End-to-End)

Note: Phi-4 requires additional merge-lora step. Please follow the steps. 

```bash
# 1. Export (x86 host)
tensorrt-edgellm-export-llm --model_dir Qwen/Qwen2.5-VL-3B-Instruct --output_dir onnx_models/qwen2.5-vl-3b
tensorrt-edgellm-export-visual --model_dir Qwen/Qwen2.5-VL-3B-Instruct --output_dir onnx_models/qwen2.5-vl-3b/visual_enc_onnx

# 2. Build Engines (Thor device)
./build/examples/llm/llm_build --onnxDir onnx_models/qwen2.5-vl-3b --engineDir engines/qwen2.5-vl-3b --vlm
./build/examples/multimodal/visual_build --onnxDir onnx_models/qwen2.5-vl-3b/visual_enc_onnx --engineDir visual_engines/qwen2.5-vl-3b

# 3. Run Inference (Thor device)
./build/examples/llm/llm_inference --engineDir engines/qwen2.5-vl-3b --multimodalEngineDir visual_engines/qwen2.5-vl-3b --inputFile input_with_images.json --outputFile output.json
```

### Phi-4 and Multimodal VLM with LoRA (End-to-End)
**NOTE: LoRA model is not compatible with the quantization pipeline, thus we need to merge the lora adapter into main model at first.**
```bash
# 0. Clone Phi-4-multimodal-instruct into disk
git clone https://huggingface.co/microsoft/Phi-4-multimodal-instruct
cd Phi-4-multimodal-instruct
git lfs pull

# 1. Merge LoRA (x86 host)
tensorrt-edgellm-merge-lora --model_dir Phi-4-multimodal-instruct \
                            --lora_dir Phi-4-multimodal-instruct/vision-lora \
                            --output_dir Phi-4-multimodal-instruct-merged-vision

# 2. Quantize (x86 host)
tensorrt-edgellm-quantize-llm --model_dir Phi-4-multimodal-instruct-merged-vision \
                               --output_dir Phi-4-multimodal-instruct-merged-vision-nvfp4 \
                               --quantization=nvfp4

# 3. Export (x86 host)
tensorrt-edgellm-export-llm --model_dir Phi-4-multimodal-instruct-merged-vision-nvfp4 --output_dir onnx_models/phi4-mm
# Use the original weights for visual model export
tensorrt-edgellm-export-visual --model_dir Phi-4-multimodal-instruct --output_dir onnx_models/phi4-mm/visual_enc_onnx

# 4. Build Engines (Thor device)
./build/examples/llm/llm_build --onnxDir onnx_models/phi4-mm --engineDir engines/phi4-mm --vlm
./build/examples/multimodal/visual_build --onnxDir onnx_models/phi4-mm/visual_enc_onnx --engineDir visual_engines/phi4-mm

# 5. Run Inference (Thor device)
./build/examples/llm/llm_inference --engineDir engines/phi4-mm --multimodalEngineDir visual_engines/phi4-mm --inputFile input_with_images.json --outputFile output.json
```

### Multimodal VLM with EAGLE Speculative Decoding (End-to-End)

```bash
# 1. Export (x86 host)
tensorrt-edgellm-export-llm --model_dir Qwen/Qwen2.5-VL-7B-Instruct --output_dir onnx_models/qwen2.5-vl-7b_eagle_base --is_eagle_base
tensorrt-edgellm-export-draft --base_model_dir Qwen/Qwen2.5-VL-7B-Instruct --draft_model_dir path/to/draft --output_dir onnx_models/qwen2.5-vl-7b_eagle_draft --use_prompt_tuning
tensorrt-edgellm-export-visual --model_dir Qwen/Qwen2.5-VL-7B-Instruct --output_dir onnx_models/qwen2.5-vl-7b/visual_enc_onnx

# 2. Build Engines (Thor device)
./build/examples/llm/llm_build --onnxDir onnx_models/qwen2.5-vl-7b_eagle_base --engineDir engines/qwen2.5-vl-7b_eagle --vlm --eagleBase
./build/examples/llm/llm_build --onnxDir onnx_models/qwen2.5-vl-7b_eagle_draft --engineDir engines/qwen2.5-vl-7b_eagle --vlm --eagleDraft
./build/examples/multimodal/visual_build --onnxDir onnx_models/qwen2.5-vl-7b/visual_enc_onnx --engineDir visual_engines/qwen2.5-vl-7b

# 3. Run Inference (Thor device)
./build/examples/llm/llm_inference --engineDir engines/qwen2.5-vl-7b_eagle --multimodalEngineDir visual_engines/qwen2.5-vl-7b --inputFile input_with_images.json --outputFile output.json --eagle
```

---

## Input File Formats

### VLM Input Format (`input_with_images.json`)

For multimodal (VLM) models, create an input JSON file with image content:

```json
{
    "batch_size": 1,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
    "max_generate_length": 128,
    "requests": [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": "examples/multimodal/pics/woman_and_dog.jpeg"
                        },
                        {
                            "type": "text",
                            "text": "Please describe the image."
                        }
                    ]
                }
            ]
        }
    ]
}
```

### LLM Input Format (`input.json`)

For standard LLM models (text-only), refer to `examples/llm/INPUT_FORMAT.md`.

---

## Common Parameters

### Build Parameters (`llm_build`, `visual_build`)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--onnxDir` | Input ONNX directory | Required |
| `--engineDir` | Output engine directory | Required |
| `--maxBatchSize` | Maximum batch size | 4 |
| `--maxInputLen` | Maximum input length | 128 |
| `--maxKVCacheCapacity` | Maximum KV-cache capacity | 4096 |
| `--vlm` | VLM mode | false |
| `--eagleBase/Draft` | EAGLE mode | false |
| `--maxLoraRank` | LoRA rank (0=disabled) | 0 |

### Inference Parameters (`llm_inference`)

| Parameter | Description |
|-----------|-------------|
| `--engineDir` | Engine directory (required) |
| `--multimodalEngineDir` | Visual/draft engine (for VLM/EAGLE) |
| `--inputFile` | Input JSON path (required) |
| `--outputFile` | Output JSON path (required) |
| `--eagle` | Enable EAGLE mode |
| `--dumpProfile` | Enable profiling |
| `--profileOutputFile` | Profile output path |

**Note:** Sampling parameters (temperature, top_p, top_k) go in the input JSON. Refer to `examples/llm/INPUT_FORMAT.md`.

---

## Next Steps

Now that you've explored the examples:

1. **Customize for Your Needs**: Learn how to extend and customize the framework in the [Customization Guide](07_Customization_Guide.md)
2. **Build Your Application**: Use the examples as templates for your own applications
3. **Optimize Performance**: Experiment with different quantization methods, batch sizes, and CUDA graphs

---

## Additional Resources

- [Overview](01.1_Overview.md)
- [Quick Start Guide](01.2_Quick_Start_Guide.md)
- [Supported Models](02_Supported_Models.md)
- [Customization Guide](07_Customization_Guide.md)
