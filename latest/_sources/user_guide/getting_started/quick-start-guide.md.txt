# Quick Start

This guide runs the image-capable
[Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) checkpoint through
CPU ONNX export, TensorRT engine build, C++ vision-language inference, and the
OpenAI-compatible server. Complete [Installation](installation.md) first.

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace/Qwen3.5-0.8B
mkdir -p "$WORKSPACE_DIR"
```

## 1. Export the Checkpoint

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

## 2. Build the Engines

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

## 3. Run Vision-Language Inference

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

## 4. Start the OpenAI-Compatible Server

The server uses the same engines built above. It requires the `server` package
extra and the C++ Python binding. From the repository root, install the extra
in the active Edge-LLM environment and reconfigure the existing build directory
without changing its platform settings:

```bash
python -m pip install -e ".[server]"

cmake -S . -B build \
  -DBUILD_PYTHON_BINDINGS=ON \
  -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')"
cmake --build build --target _edgellm_runtime -j$(nproc)
```

Start the server from the repository root. Local media is disabled by default;
the final option grants access only to the example-image directory.

```bash
python -m experimental.server \
  --model "$WORKSPACE_DIR/engines/llm" \
  --multimodal-engine-dir "$WORKSPACE_DIR/engines" \
  --allowed-local-media-path "$PWD/examples/multimodal/pics" \
  --host 127.0.0.1 \
  --port 8000
```

In another terminal, run the request from the repository root:

```bash
IMAGE_PATH=$(realpath examples/multimodal/pics/red_panda.jpeg)

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data-binary @- <<EOF
{
  "model": "Qwen3.5-0.8B",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "$IMAGE_PATH"},
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
checkpoint-to-engine frontend.
