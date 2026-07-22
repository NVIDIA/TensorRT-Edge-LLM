# Quick Start Guide

> **Repository:** [github.com/NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)

This quick start guide will get you up and running with TensorRT Edge-LLM in ~15 minutes.

**Prerequisites:** Complete the [Installation Guide](installation.md) on the machine where you will run TensorRT Edge-LLM. For the manual export and device-transfer path, set up both the x86 host and edge device.

---

## Recommended: High-Level API or Server

For Jetson Thor, DGX Spark, and x86 development, use the high-level Python API or the OpenAI-compatible server. Build the project once with Python bindings enabled, then let the high-level Python API export, build, load, and run the model from a HuggingFace checkpoint.

Install the server dependencies before configuring CMake with Python bindings:

```bash
cd /path/to/TensorRT-Edge-LLM
pip install -r requirements-server.txt
```

For x86 development:

```bash
cd /path/to/TensorRT-Edge-LLM

mkdir -p build
cd build
cmake .. \
  -DTRT_PACKAGE_DIR=$TRT_PACKAGE_DIR \
  -DCUDA_CTK_VERSION=<YOUR_CUDA_VERSION> \
  -DENABLE_CUTE_DSL=ALL \
  -DBUILD_PYTHON_BINDINGS=ON
make -j$(nproc)
cd ..
```

For Jetson Thor on JetPack 7.2, follow the CMake pattern used by CI and enable
CuTe DSL prebuilt kernels. For JetPack 7.0/7.1, use
`-DCUDA_CTK_VERSION=13.0` instead.

```bash
cd /path/to/TensorRT-Edge-LLM

mkdir -p build
cd build
cmake .. \
  -DTRT_PACKAGE_DIR=/usr \
  -DCUDA_CTK_VERSION=13.2 \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-thor \
  -DENABLE_CUTE_DSL=ALL \
  -DBUILD_PYTHON_BINDINGS=ON
make -j$(nproc)
cd ..
```

For DGX Spark, use `gb10` as the embedded target and CUDA Toolkit 13.0:

```bash
cd /path/to/TensorRT-Edge-LLM

mkdir -p build
cd build
cmake .. \
  -DTRT_PACKAGE_DIR=/usr \
  -DCUDA_CTK_VERSION=13.0 \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=gb10 \
  -DENABLE_CUTE_DSL=ALL \
  -DBUILD_PYTHON_BINDINGS=ON
make -j$(nproc)
cd ..
```

After either build:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

Run a prompt with the high-level Python API:

```bash
python - <<'PY'
from experimental.server import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B")
outputs = llm.generate(
    ["What is the capital of the United States?"],
    SamplingParams(max_tokens=128),
)
print(outputs[0].text)
PY
```

Or launch an OpenAI-compatible server:

```bash
python -m experimental.server \
  --model Qwen/Qwen3-0.6B \
  --port 8000
```

Query the server from another terminal:

```bash
curl -sN http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is the capital of the United States?"}], "max_tokens": 128}'
```

For more options, including loading existing ONNX or engine directories, see [Experimental High-Level Python API and Server](../examples/experimental-server.md).

---

## Manual Export and C++ Runtime Path

Use this path when you need explicit control over ONNX export, engine build flags, file transfer, or the low-level C++ examples.

### Part 1: Export on x86 Host

This part runs on a standard x86 host with an NVIDIA GPU. **DriveOS users:** this process does not need to run in DriveOS Docker; use your regular x86 development machine.

#### Recommended: Checkpoint Exporter

The checkpoint exporter (`tensorrt_edgellm`) is the recommended export path. It reads FP16/BF16 and pre-quantized HuggingFace checkpoints directly and exports to ONNX in a single command. If you need to create a quantized checkpoint from an FP16/BF16 source model, install the standalone quantization requirements from the Installation Guide, run `tensorrt-edgellm-quantize`, and then export the generated checkpoint with `tensorrt_edgellm`.

Let's use [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) as a lightweight example:

```bash
# Set up workspace directory
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3-0.6B
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# Set PYTHONPATH to include the quantization package and loader
export PYTHONPATH=/path/to/TensorRT-Edge-LLM:$PYTHONPATH

# Export FP16 model to ONNX
tensorrt-edgellm-export \
    Qwen/Qwen3-0.6B \
    $MODEL_NAME/onnx

# Or quantize an FP16/BF16 model first, then export the quantized checkpoint
tensorrt-edgellm-quantize llm \
    --model_dir Qwen/Qwen3-0.6B \
    --output_dir $MODEL_NAME/quantized \
    --quantization nvfp4 \
    --lm_head_quantization nvfp4

tensorrt-edgellm-export \
    $MODEL_NAME/quantized \
    $MODEL_NAME/onnx

# Or export a pre-quantized NVFP4 checkpoint (no separate quantization step)
# See https://huggingface.co/nvidia/Qwen3-8B-NVFP4
tensorrt-edgellm-export \
    /path/to/nvidia/Qwen3-8B-NVFP4 \
    Qwen3-8B-NVFP4/onnx

# Or export a pre-quantized INT4-AWQ checkpoint
# See https://huggingface.co/Qwen/Qwen3-4B-AWQ
tensorrt-edgellm-export \
    /path/to/Qwen/Qwen3-4B-AWQ \
    Qwen3-4B-AWQ/onnx \
    --externalize-weights int4_ffn
```

For pre-quantized checkpoints (FP8, INT4 AWQ/GPTQ, NVFP4), simply point the loader at the quantized checkpoint directory. For quantization options, FP8 KV cache, FP8 embedding, LoRA, and vocabulary reduction, see [Quantization](../features/quantization.md), [FP8 KV Cache](../features/FP8KV.md), [FP8 Embedding](../features/fp8-embedding.md), [LoRA](../features/lora.md), and [Vocabulary Reduction](../features/reduce-vocab.md).

For INT4 engine builds on Jetson Orin devices with less system memory, such as
Jetson Orin Nano, externalized weights are recommended to reduce engine build
memory. Use
`--externalize-weights int4_ffn` for dense INT4 checkpoints and
`--externalize-weights int4_ffn int4_moe` for INT4 MoE checkpoints.

#### Transfer to Device

Transfer the ONNX folder to your Thor device:

```bash
# From x86 host - transfer to device
scp -r $MODEL_NAME/onnx <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

> **Note:** Replace `<device_user>` and `<device_ip>` with your actual device credentials (e.g., `nvidia@192.168.1.100`). If the directory doesn't exist on the device, create it first: `ssh <device_user>@<device_ip> "mkdir -p ~/tensorrt-edgellm-workspace/$MODEL_NAME"`

---

### Part 2: Build and Run on Edge Device

#### Build TensorRT Engine

On your Thor device:

```bash
# Set up workspace directory
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Qwen3-0.6B
cd /path/to/TensorRT-Edge-LLM

# Build engine
./build/examples/llm/llm_build \
    --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/llm \
    --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
    --maxBatchSize 1 \
    --maxInputLen 1024 \
    --maxKVCacheCapacity 4096
```

Build time: ~2-5 minutes

#### Run Inference

Create an input file with a sample question:

```bash
cat > $WORKSPACE_DIR/input.json << 'EOF'
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
                    "role": "user",
                    "content": "What is the capital of United States?"
                }
            ]
        }
    ]
}
EOF
```

> **Tip:** You can also use example input files from `/path/to/TensorRT-Edge-LLM/tests/test_cases/` (e.g., `llm_basic.json`) instead of creating your own.

Run inference:

```bash
cd /path/to/TensorRT-Edge-LLM

./build/examples/llm/llm_inference \
    --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
    --inputFile $WORKSPACE_DIR/input.json \
    --outputFile $WORKSPACE_DIR/output.json
```

Verify the output:

```bash
# View the model response
cat $WORKSPACE_DIR/output.json
```

You should see a JSON response with the model's answer, similar to:

```json
{
  "responses": [
    {
      "output_text": "The capital of the United States is Washington, D.C.",
      "request_idx": 0,
      "batch_idx": 0,
      "finish_reason": "stop",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "What is the capital of the United States?"
            }
          ]
        }
      ],
      "formatted_system_prompt": "",
      "formatted_complete_request": "..."
    }
  ]
}
```

### Inference and Benchmarking Tools

TensorRT Edge-LLM provides two LLM runtime examples for different purposes:

- `llm_inference` is the end-to-end inference example. Use it when you want to run real JSON requests, apply chat templates, generate text, write response JSON, and validate application-level behavior. It can also report overall performance metrics such as tokens/sec.
- `llm_bench` is the benchmark example. Use it when you want synthetic prefill/decode timing for a built engine without preparing an input JSON file. By default it reports overall E2E timing; pass `--profile` to collect per-layer profiling for kernel-level breakdowns.

For example, benchmark prefill latency for the engine built above:

```bash
./build/examples/llm/llm_bench \
    --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
    --mode prefill \
    --inputLen 128 \
    --batchSize 1
```

To collect layer-level profiling in addition to the benchmark summary, add `--profile`.

**Success!** 🎉 You've successfully run LLM inference on your edge device!

---

## Next Steps

**For more advanced workflows, see the example guides:**
- **[VLM Inference](../examples/vlm.md)** - Vision-language models with image understanding
- **[Speculative Decoding](../examples/speculative-decoding.md)** - EAGLE3, MTP, and DFlash speculative decoding workflows
- **[Phi-4-Multimodal](../examples/phi4.md)** - Phi-4 Multimodal
- **[ASR](../examples/asr.md)** - Automatic speech recognition
- **[MoE](../examples/moe.md)** - Mixture of Experts models (CPU-only export, Qwen3-30B-A3B-GPTQ-Int4/NVFP4)
- **[TTS](../examples/tts.md)** - Text-to-speech synthesis
- **[Alpamayo-R1-10B](../examples/vla.md)** - Vision-language-action inference with trajectory prediction

**Checkpoint Exporter:** For detailed documentation on the export pipeline and pre-quantized checkpoint support, see [Checkpoint Exporter](../../developer_guide/software-design/checkpoint-export.md).

**Quantization:** To create quantized checkpoints for `tensorrt_edgellm`, see [Quantization](../features/quantization.md).

**Experimental Python API and Server:** To use the vLLM-style high-level Python API or an OpenAI-compatible chat server, see [Experimental High-Level Python API and Server](../examples/experimental-server.md).

**Input Format:** Our format matches closely with the OpenAI API format. See [Input Format Guide](../format/input-format.md) for detailed specifications. Example input files are available in `tests/test_cases/` (e.g., `llm_basic.json`, `vlm_basic.json`).
