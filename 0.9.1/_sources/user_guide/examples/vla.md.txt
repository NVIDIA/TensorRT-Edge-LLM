# Alpamayo-R1-10B (Vision-Language-Action)

Complete workflow for running [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B), a Vision-Language-Action (VLA) model that combines a VLM backbone with an action expert.

**Currently supported model:** [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B)

> **Prerequisites:** Complete the [Installation Guide](../getting_started/installation.md) for both x86 host and edge device before proceeding. The Alpamayo 1 checkpoint may require Hugging Face login and license access before downloading.

---

## Architecture Overview

Alpamayo-R1-10B runs as a chained VLM + action pipeline:

1. **VLM backbone** — exported through `tensorrt_edgellm` as `onnx/llm` and `onnx/visual`; it processes image, text, and trajectory-history tokens and generates language output.
2. **Action expert** — exported through `tensorrt_edgellm` as `onnx/action`; it consumes the VLM KV cache and produces future action waypoints.

Both stages are run with a single `action_inference` invocation. The engines are built separately and loaded together at runtime.

The current TensorRT Edge-LLM input parser accepts image files, text, and
trajectory history points as `[x, y, z]`. The action output is written as
`output_trajectory`, a list of `(accel, kappa)` acceleration and curvature pairs.

---

## Step 1: Export (x86 Host)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Alpamayo-R1-10B
mkdir -p $WORKSPACE_DIR
cd $WORKSPACE_DIR

# Download Alpamayo-R1-10B from HuggingFace
hf auth login
hf download nvidia/Alpamayo-R1-10B --local-dir $MODEL_NAME

# Export language model backbone, visual encoder, and action expert
tensorrt-edgellm-export \
  $MODEL_NAME \
  $MODEL_NAME/onnx \
  --max-kv-cache-capacity 4096
```

This creates `onnx/llm`, `onnx/visual`, and `onnx/action`.

> **Note:** Only FP16 is supported for Alpamayo export in this release. The `--max-kv-cache-capacity` value must match the `--maxKVCacheCapacity` used when building the LLM engine in Step 3.

## Step 2: Transfer to Device

```bash
scp -r $MODEL_NAME/onnx \
  <device_user>@<device_ip>:~/tensorrt-edgellm-workspace/$MODEL_NAME/
```

## Step 3: Build Engines (Thor Device)

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace
export MODEL_NAME=Alpamayo-R1-10B
cd /path/to/TensorRT-Edge-LLM

# Build language model engine
./build/examples/llm/llm_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/llm \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --maxInputLen 3424 \
  --maxKVCacheCapacity 4096 \
  --maxBatchSize 6

# Build visual encoder engine
./build/examples/multimodal/visual_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/visual \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --minImageTokens 160 \
  --maxImageTokens 18432 \
  --maxImageTokensPerImage 192

# Build action expert engine
./build/examples/multimodal/action_build \
  --onnxDir $WORKSPACE_DIR/$MODEL_NAME/onnx/action \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --maxBatchSize 6
```

`visual_build` writes `$WORKSPACE_DIR/$MODEL_NAME/engines/visual/`, and
`action_build` writes `$WORKSPACE_DIR/$MODEL_NAME/engines/action/`. Both
subdirectories are loaded through `--multimodalEngineDir` during inference.

## Step 4: Run Inference (Thor Device)

Create an input file `$WORKSPACE_DIR/input_action.json`. The `trajectory`
content item provides past egomotion history as `[x, y, z]` positions. Replace
the image paths with actual RGB image files. The checkpoint card describes the
primary setting as 4 cameras over 4 timesteps; the runtime parser accepts the
images as an ordered list of image content items.

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
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a driving assistant that generates safe and accurate actions."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "/path/to/frame_00.png"},
                        {"type": "image", "image": "/path/to/frame_01.png"},
                        {"type": "image", "image": "/path/to/frame_02.png"},
                        {"type": "image", "image": "/path/to/frame_03.png"},
                        {"type": "image", "image": "/path/to/frame_04.png"},
                        {"type": "image", "image": "/path/to/frame_05.png"},
                        {"type": "image", "image": "/path/to/frame_06.png"},
                        {"type": "image", "image": "/path/to/frame_07.png"},
                        {"type": "image", "image": "/path/to/frame_08.png"},
                        {"type": "image", "image": "/path/to/frame_09.png"},
                        {"type": "image", "image": "/path/to/frame_10.png"},
                        {"type": "image", "image": "/path/to/frame_11.png"},
                        {"type": "image", "image": "/path/to/frame_12.png"},
                        {"type": "image", "image": "/path/to/frame_13.png"},
                        {"type": "image", "image": "/path/to/frame_14.png"},
                        {"type": "image", "image": "/path/to/frame_15.png"},
                        {
                            "type": "trajectory",
                            "trajectory": [
                                [-13.570470809936523, 0.06137955188751221, -0.021795138716697693],
                                [-12.617349624633789, 0.05502257123589516, -0.021485785022377968],
                                [-11.672354698181152, 0.04983367770910263, -0.019761288538575172],
                                [-10.733928680419922, 0.04469458386301994, -0.01891801320016384],
                                [-9.802403450012207, 0.03998061269521713, -0.019592544063925743],
                                [-8.87903118133545, 0.034502264112234116, -0.016824787482619286],
                                [-7.963467121124268, 0.03007129393517971, -0.011910118162631989],
                                [-7.054953098297119, 0.02519317716360092, -0.008388019166886806],
                                [-6.15440559387207, 0.020703254267573357, -0.003812047652900219],
                                [-5.261260032653809, 0.016889382153749466, 0.0007052596774883568],
                                [-4.374241828918457, 0.012481293641030788, 0.004680476617068052],
                                [-3.4911155700683594, 0.00914590060710907, 0.0029235705733299255],
                                [-2.611931085586548, 0.005678099580109119, 7.79956899350509e-05],
                                [-1.736931562423706, 0.003552841255441308, -0.0018224255181849003],
                                [-0.8664319515228271, 0.001968142343685031, -0.001738768769428134],
                                [0.0, 0.0, 0.0]
                            ]
                        },
                        {
                            "type": "text",
                            "text": "output the chain-of-thought reasoning of the driving process, then output the future trajectory."
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

./build/examples/multimodal/action_inference \
  --engineDir $WORKSPACE_DIR/$MODEL_NAME/engines/llm \
  --multimodalEngineDir $WORKSPACE_DIR/$MODEL_NAME/engines \
  --inputFile $WORKSPACE_DIR/input_action.json \
  --outputFile $WORKSPACE_DIR/output_action.json
```

Check `output_action.json` for the response. Each entry includes `output_text` (VLM language output) and `output_trajectory` (predicted (accel, kappa) acceleration and curvature pairs).
