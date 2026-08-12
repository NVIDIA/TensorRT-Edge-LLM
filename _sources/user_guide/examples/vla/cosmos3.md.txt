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
tensorrt-edgellm-export \
  "$POLICY_CHECKPOINT" \
  "$ONNX_DIR" \
  --task policy
```

### 2. Build All Policy Engines

```bash
./build/experimental_models/cosmos3/examples/cosmos3_policy_build \
  --onnxDir "$ONNX_DIR" \
  --engineDir "$ENGINE_DIR"
```

Use `--maxBatchSize N` when the runtime must accept more than one prompt.

### 3. Run the Policy

For one observation image:

```bash
./build/experimental_models/cosmos3/examples/cosmos3_policy_inference \
  --engineDir "$ENGINE_DIR" \
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
tensorrt-edgellm-export \
  "$REASONING_CHECKPOINT" \
  "$ONNX_DIR/reasoning" \
  --task reasoning
```

### 2. Build the LLM and Visual Engines

```bash
./build/examples/llm/llm_build \
  --onnxDir "$ONNX_DIR/reasoning/llm" \
  --engineDir "$ENGINE_DIR/reasoning" \
  --maxInputLen 2048 \
  --maxKVCacheCapacity 4096

./build/examples/multimodal/visual_build \
  --onnxDir "$ONNX_DIR/reasoning/visual" \
  --engineDir "$ENGINE_DIR/reasoning"
```

### 3. Run Reasoning

Create an input file using the standard
[image message format](../../format/input-format.md#message-fields), then run:

```bash
./build/examples/llm/llm_inference \
  --engineDir "$ENGINE_DIR/reasoning" \
  --multimodalEngineDir "$ENGINE_DIR/reasoning" \
  --inputFile input.json \
  --outputFile output.json
```

See the [Cosmos3 model guide](../../../developer_guide/models/cosmos3.md) for
the component contracts and implementation details.
