# MoE (Mixture of Experts)

Complete workflow for Mixture of Experts (MoE) models using pre-quantized INT4 or NVFP4 checkpoints.

> **Note:** For very large NVFP4 MoE checkpoints such as Nemotron Super 120B,
> externalize NVFP4 MoE plugin weights during export and keep the generated
> safetensors file with the ONNX directory.

> **Note:** NVFP4 MoE uses separate plugins with different FC1 weight layouts:
> `Nvfp4MoePlugin` on SM100/101/110 (default) and `NvFP4MoEPluginGeforce` on
> SM120/121. Set `EDGELLM_NVFP4_MOE_TARGET=sm12x` before export for consumer
> Blackwell (including DGX Spark GB10); re-export if you change deployment GPU.

> **Prerequisites:** Complete the [Installation Guide](../getting_started/installation.md) before proceeding.

---

## Step 1: Export (x86 Host, CPU-only)

Export always runs on CPU; no GPU is required:

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3-30B-A3B-GPTQ-Int4
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

tensorrt-edgellm-export \
  Qwen/Qwen3-30B-A3B-GPTQ-Int4 \
  $MODEL_NAME/exported

mkdir -p $MODEL_NAME/onnx
cp -a $MODEL_NAME/exported/llm/. $MODEL_NAME/onnx/
```

For Nemotron Super 120B NVFP4, externalize the MoE plugin weights to reduce engine build memory pressure:

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# For NVFP4 MoE engines on SM120/SM121, set the target before exporting:
export EDGELLM_NVFP4_MOE_TARGET=sm12x

tensorrt-edgellm-export \
  nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  $MODEL_NAME/exported \
  --externalize-weights nvfp4_moe \
  --max-kv-cache-capacity 4096

mkdir -p $MODEL_NAME/onnx
cp -a $MODEL_NAME/exported/llm/. $MODEL_NAME/onnx/
```

## Step 2: Transfer to Device

```bash
# Transfer ONNX to device
scp -r $MODEL_NAME/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

## Step 3: Build Engine (Edge Device)

`llm_build` is the same as for regular LLMs:

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3-30B-A3B-GPTQ-Int4
cd /path/to/TensorRT-Edge-LLM

./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --maxBatchSize 1 \
  --maxInputLen 3072 \
  --maxKVCacheCapacity 4096
```

## Step 4: Run Inference (Edge Device)

`llm_inference` is the same as for regular LLMs:

```bash
cd /path/to/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --inputFile $WORKSPACE_DIR/input.json \
  --outputFile $WORKSPACE_DIR/output.json
```

MoE export uses CPU-only; build and inference use the same `llm_build` and `llm_inference` as standard LLMs.
