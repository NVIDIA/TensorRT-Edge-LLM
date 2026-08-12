# Phi-4-Multimodal

Phi-4-Multimodal requires merging its vision LoRA adapter into the base model before quantization and export.

**Example model:** Phi-4-multimodal-instruct

> **Prerequisites:** Complete the [Installation Guide](../getting_started/installation.md) before proceeding.

---

## Step 1: Merge, Quantize, and Export (x86 Host)

```bash
cd /path/to/TensorRT-Edge-LLM
pip3 install ".[tools]"

export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Phi-4-multimodal-instruct
export MODEL_DIR=$WORKSPACE_DIR/$MODEL_NAME
export HF_MODEL_DIR=$MODEL_DIR/hf
mkdir -p $MODEL_DIR
cd $WORKSPACE_DIR

# Clone Phi-4-multimodal-instruct from HuggingFace
git lfs install
git clone https://huggingface.co/microsoft/Phi-4-multimodal-instruct $HF_MODEL_DIR
git -C $HF_MODEL_DIR lfs pull

# Merge vision LoRA adapter into base model
tensorrt-edgellm-merge-lora \
  --model_dir $HF_MODEL_DIR \
  --lora_dir $HF_MODEL_DIR/vision-lora \
  --output_dir $MODEL_DIR/merged

# Quantize merged model
tensorrt-edgellm-quantize llm \
  --model_dir $MODEL_DIR/merged \
  --output_dir $MODEL_DIR/quantized \
  --quantization nvfp4 \
  --lm_head_quantization nvfp4

# Export language model and visual encoder
tensorrt-edgellm-export \
  $MODEL_DIR/quantized \
  $MODEL_DIR/onnx
```

## Step 2: Transfer to Device

```bash
# Transfer ONNX to device
scp -r $MODEL_DIR/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

## Step 3: Build Engines (Thor Device)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Phi-4-multimodal-instruct
cd /path/to/TensorRT-Edge-LLM

# Build language model engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/llm \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --maxBatchSize 1 \
  --maxInputLen 1536 \
  --maxKVCacheCapacity 2048

# Build visual encoder engine
./build/examples/multimodal/visual_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/visual \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --minImageTokens 256 \
  --maxImageTokens 1024 \
  --maxImageTokensPerImage 512
```

## Step 4: Run Inference (Thor Device)

```bash
cd /path/to/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --multimodalEngineDir $WORKSPACE_DIR/$MODEL_NAME/engines/visual \
  --inputFile $WORKSPACE_DIR/input.json \
  --outputFile $WORKSPACE_DIR/output.json
```
