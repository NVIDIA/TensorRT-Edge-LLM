# Cosmos3-Edge

This guide covers policy generation and multimodal reasoning with
[nvidia/Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge).

Complete [Installation](../../getting_started/installation.md) first. Policy
generation also requires `-DBUILD_EXPERIMENTAL_MODELS=ON`.

```bash
export POLICY_CHECKPOINT=nvidia/Cosmos3-Edge-Policy-DROID
export REASONING_CHECKPOINT=nvidia/Cosmos3-Edge
export ONNX_DIR=$HOME/tensorrt-edgellm-workspace/Cosmos3-Edge/onnx
export ENGINE_DIR=$HOME/tensorrt-edgellm-workspace/Cosmos3-Edge/engines
```

## Policy Generation

### 1. Export the Policy on CPU

```bash
trtedge export \
  "$POLICY_CHECKPOINT" \
  "$ONNX_DIR" \
  --task policy
```

### 2. Build All Policy Engines

```bash
./build/experimental_models/cosmos3/examples/cosmos3_policy_build \
  --onnx-dir "$ONNX_DIR" \
  --engine-dir "$ENGINE_DIR"
```

Use `--max-batch-size N` when the runtime must accept more than one prompt.

### 3. Run the Policy

For one observation image:

```bash
./build/experimental_models/cosmos3/examples/cosmos3_policy_inference \
  --engine-dir "$ENGINE_DIR" \
  --image observation.png \
  --prompt "Pick up the banana and place it in the bowl." \
  --output action.json
```

For an ordered frame list, replace `--image` with:

```bash
--video frame_00.png,frame_01.png,frame_02.png
```

The current policy conditions on the most recent frame. The JSON output reports
the action tensor, shape `[batch, chunk, action_dimension]`, policy domain,
denoise-step count, and whether all values are finite. `--steps` selects the
denoise-step count and `--seed` controls deterministic initial noise.

## Multimodal Reasoning

### 1. Export the Reasoner on CPU

```bash
trtedge export \
  "$REASONING_CHECKPOINT" \
  "$ONNX_DIR/reasoning" \
  --task reasoning
```

### 2. Build the LLM and Visual Engines

```bash
./build/examples/llm/llm_build \
  --onnx-dir "$ONNX_DIR/reasoning/llm" \
  --engine-dir "$ENGINE_DIR/reasoning" \
  --max-input-len 2048 \
  --max-kv-cache-capacity 4096

./build/examples/multimodal/visual_build \
  --onnx-dir "$ONNX_DIR/reasoning/visual" \
  --engine-dir "$ENGINE_DIR/reasoning"
```

### 3. Run Reasoning

For image reasoning, create an input file using the standard
[image message format](../../format/input-format.md#message-fields), then run:

```bash
./build/examples/llm/llm_inference \
  --engine-dir "$ENGINE_DIR/reasoning" \
  --multimodal-engine-dir "$ENGINE_DIR/reasoning" \
  --input-file input.json \
  --output-file output.json
```

For native video-file input, launch the OpenAI-compatible server:

```bash
trtedge serve "$REASONING_CHECKPOINT" \
  --cache-dir "$HOME/tensorrt-edgellm-workspace/cache" \
  --max-image-tokens 4096 \
  --max-image-tokens-per-image 4096 \
  --allowed-local-media-path /data/media
```

Then send the video itself, rather than converting it to image content blocks:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/Cosmos3-Edge",
    "messages": [{
      "role": "user",
      "content": [
        {
          "type": "video_url",
          "video_url": {"url": "file:///data/media/example.mp4"},
          "fps": 2.0
        },
        {"type": "text", "text": "Describe the actions in this video."}
      ]
    }],
    "max_tokens": 256
  }'
```

The server decodes and samples the clip to the visual engine profile and keeps
it as one video input through the native runtime.

See the [Cosmos3 model guide](../../../developer_guide/models/cosmos3.md) for
the component contracts and implementation details.
