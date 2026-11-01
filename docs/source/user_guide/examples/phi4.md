# Phi-4-Multimodal

Phi-4-Multimodal requires merging its vision LoRA adapter into the base model before quantization and export.

**Example model:** Phi-4-multimodal-instruct

> **Prerequisites:** Complete the [Installation Guide](../getting_started/installation.md) before proceeding.

---

## Step 1: Merge, Quantize, and Export (x86 Host)

Install the `tools` extra from a [published wheel](../getting_started/installation.md#optional-python-dependencies)
or the source checkout. The later C++ engine-build and inference steps require
the source workflow.

```bash
cd /path/to/TensorRT-Edge-LLM

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
trtedge merge-lora \
  --model-dir $HF_MODEL_DIR \
  --lora-dir $HF_MODEL_DIR/vision-lora \
  --output-dir $MODEL_DIR/merged

# Quantize merged model
trtedge quantize llm \
  --model-dir $MODEL_DIR/merged \
  --output-dir $MODEL_DIR/quantized \
  --quantization nvfp4 \
  --lm-head-quantization nvfp4

# Export language model and visual encoder
trtedge export \
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
  --onnx-dir $WORKSPACE_DIR/$MODEL_NAME/onnx/llm \
  --engine-dir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --max-batch-size 1 \
  --max-input-len 1536 \
  --max-kv-cache-capacity 2048

# Build visual encoder engine
./build/examples/multimodal/visual_build \
  --onnx-dir $WORKSPACE_DIR/$MODEL_NAME/onnx/visual \
  --engine-dir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --min-image-tokens 256 \
  --max-image-tokens 1024 \
  --max-image-tokens-per-image 512
```

## Step 4: Run Inference (Thor Device)

```bash
cd /path/to/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engine-dir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --multimodal-engine-dir $WORKSPACE_DIR/$MODEL_NAME/engines/visual \
  --input-file $WORKSPACE_DIR/input.json \
  --output-file $WORKSPACE_DIR/output.json
```
