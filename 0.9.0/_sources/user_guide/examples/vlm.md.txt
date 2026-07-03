# VLM (Vision-Language Model) Inference

Complete workflow for vision-language models with image understanding capabilities.

**Example model:** Qwen2.5-VL-3B-Instruct

> **Note:** For Phi-4-Multimodal, which requires a LoRA merge step before export, see the [Phi-4-Multimodal Guide](phi4.md).

> **Prerequisites:** Complete the [Installation Guide](../getting_started/installation.md) for both x86 host and edge device before proceeding.

---

## Step 1: Quantize and Export (x86 Host)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen2.5-VL-3B-Instruct
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# Quantize the LLM weights to a unified checkpoint
tensorrt-edgellm-quantize llm \
  --model_dir Qwen/Qwen2.5-VL-3B-Instruct \
  --quantization fp8 \
  --output_dir $MODEL_NAME/quantized

# Export the language model and FP16 visual encoder
tensorrt-edgellm-export \
  $MODEL_NAME/quantized \
  $MODEL_NAME/onnx
```

To also quantize the visual tower to FP8, add `--visual_quantization fp8` and
use a multimodal calibration dataset such as `--dataset lmms-lab/MMMU`.

## Step 2: Transfer to Device

```bash
# Transfer ONNX to device
scp -r $MODEL_NAME/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

## Step 3: Build Engines (Thor Device)

```bash
# Set up workspace directory on device
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen2.5-VL-3B-Instruct
cd /path/to/TensorRT-Edge-LLM

# Build language model engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/llm \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --maxBatchSize 1 \
  --maxInputLen 1024 \
  --maxKVCacheCapacity 4096

# Build visual encoder engine
./build/examples/multimodal/visual_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/visual \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --minImageTokens 128 \
  --maxImageTokens 512 \
  --maxImageTokensPerImage 512
```

Build time: < 5 minutes

## Step 4: Run Inference (Thor Device)

Create an input file `$WORKSPACE_DIR/input_vlm.json` (replace `/path/to/image.jpg` with an actual image file path):

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
                            "image": "/path/to/image.jpg"
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

Run inference:

```bash
cd /path/to/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --multimodalEngineDir $WORKSPACE_DIR/$MODEL_NAME/engines/visual \
  --inputFile $WORKSPACE_DIR/input_vlm.json \
  --outputFile $WORKSPACE_DIR/output_vlm.json
```

Check `output_vlm.json` for vision-language model responses.
