# Quick Start

This guide provides two independent ways to run the image-capable
[Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) checkpoint:

1. CPU ONNX export, TensorRT engine build, and C++ vision-language inference.
2. Checkpoint-direct engine build and inference through the Python
   OpenAI-compatible server.

Complete [Installation](installation.md) first. You do not need to complete the
ONNX workflow before using the Python server.

Before building engines or launching inference, verify that the target has at
least the deployed model size plus 2 GB of available device memory. KV cache,
multimodal components, speculative draft engines, and larger batch or sequence
profiles can require more.

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace/Qwen3.5-0.8B
mkdir -p "$WORKSPACE_DIR"
```

## Option 1: ONNX and C++ runtime

### 1. Export the Checkpoint

Run export in the active Edge-LLM Python environment. Unquantized and supported
pre-quantized checkpoints export on CPU:

```bash
tensorrt-edgellm-export \
  Qwen/Qwen3.5-0.8B \
  "$WORKSPACE_DIR/onnx"
```

The command exports both `onnx/llm` and `onnx/visual`. If export and inference
use different machines, copy the complete output directory to the target:

```bash
rsync -a "$WORKSPACE_DIR/onnx/" \
  <user>@<target>:~/tensorrt-edgellm-workspace/Qwen3.5-0.8B/onnx/
```

Quantization is optional. Start from a supported pre-quantized checkpoint, or
run `tensorrt-edgellm-quantize` on an x86 GPU host before export. Quantization
changes model accuracy; validate a generated checkpoint against its source
model before deployment. See [Quantization](../features/quantization.md).

### 2. Build the Engines

Run both builders on the target from the repository root. Engine profile values
are deployment limits; increase them only when the workload requires it.

```bash
./build/examples/llm/llm_build \
  --onnxDir "$WORKSPACE_DIR/onnx/llm" \
  --engineDir "$WORKSPACE_DIR/engines/llm" \
  --maxBatchSize 2 \
  --maxInputLen 4096 \
  --maxKVCacheCapacity 4096

./build/examples/multimodal/visual_build \
  --onnxDir "$WORKSPACE_DIR/onnx/visual" \
  --engineDir "$WORKSPACE_DIR/engines" \
  --minImageTokens 128 \
  --maxImageTokens 4096 \
  --maxImageTokensPerImage 512
```

### 3. Run Vision-Language Inference

The repository includes the image used below. Create `$WORKSPACE_DIR/input.json`:

```json
{
  "batch_size": 1,
  "temperature": 0.0,
  "max_generate_length": 64,
  "requests": [
    {
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "image",
              "image": "examples/multimodal/pics/red_panda.jpeg"
            },
            {
              "type": "text",
              "text": "Describe this image."
            }
          ]
        }
      ]
    }
  ]
}
```

Run from the repository root so the relative image path resolves:

```bash
./build/examples/llm/llm_inference \
  --engineDir "$WORKSPACE_DIR/engines/llm" \
  --multimodalEngineDir "$WORKSPACE_DIR/engines" \
  --inputFile "$WORKSPACE_DIR/input.json" \
  --outputFile "$WORKSPACE_DIR/output.json"

cat "$WORKSPACE_DIR/output.json"
```

The response contains generated text, token IDs, token counts, and the finish
reason.

## Option 2: One-line Python server

The server does not consume the ONNX engines from Option 1. Install a
[published wheel with the `server` extra](installation.md#published-python-wheel),
or complete the [source build and server setup](installation.md#install-and-launch-the-python-server).
The wheel path does not require a source checkout or native build.

Activate that environment. For a wheel install, launch outside a source
checkout; for a source install, use the repository root. The first launch
downloads the checkpoint, builds its required components, and stores the runtime
bundle in the server cache. Set the media directory to one containing an image
you want to query:

```bash
tensorrt-edgellm-serve Qwen/Qwen3.5-0.8B --allowed-local-media-path /absolute/path/to/images
```

In another terminal, use an image inside that directory on the server machine:

```bash
IMAGE_PATH=/absolute/path/to/images/example.jpeg

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data-binary @- <<EOF
{
  "model": "Qwen/Qwen3.5-0.8B",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "file://$IMAGE_PATH"}},
        {"type": "text", "text": "Describe this image."}
      ]
    }
  ],
  "temperature": 0.0,
  "max_tokens": 64
}
EOF
```

The response follows the OpenAI chat-completions schema. See
[Experimental High-Level Python API and Server](../examples/experimental-server.md)
for streaming, batching, audio, video, and tool-calling options.

See [Input JSON Format](../format/input-format.md) for C++ request fields,
[Examples](../examples/index.md) for other model contracts, and
[Direct Engine Builder](direct-engine-builder.md) for the experimental
checkpoint-to-engine frontend. See
[Multi-Device Inference](../features/multi-device.md) for TP2 export,
per-rank engine builds, and local multi-GPU execution.
