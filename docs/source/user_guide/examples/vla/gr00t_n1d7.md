# GR00T N1.7

[GR00T N1.7](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO) is NVIDIA Isaac's flow-matching
Vision-Language-Action model. Edge-LLM runs it as three TensorRT components driven by an
experimental C++ runtime, which returns an action chunk in robot units for a set of camera views,
an instruction and a robot state. The engines come either from the ONNX-less direct builder on the
target or from a CPU ONNX export built by `gr00t_policy_build`; both routes write the same bundle. See
[GR00T N1.7 Design](../../../developer_guide/models/gr00t_n1d7.md) for the component contracts and
the reasoning behind the split.

This workflow supports the `libero_10` checkpoint's `libero_sim` embodiment, in FP16. Results are a
numerical comparison against Isaac-GR00T's own PyTorch action chunk; task success is not claimed.

| embodiment | checkpoint | state / action | horizon | cameras |
|---|---|---|---|---|
| `libero_sim` | `nvidia/GR00T-N1.7-LIBERO/libero_10` | 8 / 7 | 16 | `image`, `wrist_image`, both required |

The checkpoint names other embodiments, but every other one it equips declares relative
end-effector actions, whose conversion this runtime does not implement; the builder rejects them
rather than mis-scale a command.

Complete [Installation](../../getting_started/installation.md) and the
[ONNX-less engine build guide](../../getting_started/direct-engine-builder.md) first; GR00T also
requires `-DBUILD_EXPERIMENTAL_MODELS=ON`, and on Thor `-DENABLE_CUTE_DSL=ALL
-DCUTE_DSL_ARTIFACT_TAG=sm_110`. The `visual` engine carries a `trt_edgellm::ViTAttentionPlugin`
node; the build and inference tools load the plugin library from `EDGELLM_PLUGIN_PATH`.

The Hugging Face checkpoint ships no tokenizer: Isaac-GR00T loads its backbone's from the Hub at
run time. Both builders read it from `tokenizer/` inside the checkpoint directory instead: the
vocabulary, the chat template and special tokens, and the image preprocessor config the shared
Qwen3-VL vision runner reads. Stage the three files once:

```bash
export CHECKPOINT=/path/to/GR00T-N1.7-LIBERO/libero_10
hf download Qwen/Qwen3-VL-2B-Instruct --local-dir "$CHECKPOINT/tokenizer" \
    tokenizer.json tokenizer_config.json preprocessor_config.json
rm -rf "$CHECKPOINT/tokenizer/.cache"
```

```bash
export ENGINE_DIR=$HOME/gr00t_engines
export BUILD_DIR=/path/to/tensorrt-edge-llm/build
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
export EDGELLM_PLUGIN_PATH="$BUILD_DIR/lib/libNvInfer_edgellm_plugin.so"
```

## 1. Build all engines

### From the checkpoint, on the target

```bash
tensorrt-edgellm-build \
    --model-dir "$CHECKPOINT" --engine-dir "$ENGINE_DIR" --components all \
    --plugin-path "$EDGELLM_PLUGIN_PATH" \
    --max-input-len 1024 --max-batch-size 1
```

This writes one subdirectory per component -- `visual/`, `vl_backbone/` and `action/` -- plus the
token-embedding table and the checkpoint's tokenizer, and `action/policy.json`, the observation and
action contract: the embodiment's joint-group order and q01/q99 bounds, the camera names and the
image geometry, all resolved from the checkpoint. The runtime refuses a bundle without it, and the
build refuses a checkpoint without a loadable `tokenizer/` before it writes any engine.

`--gr00t-embodiment-id N` selects another embodiment slot; the default is resolved from the
checkpoint directory name. Every component of a bundle must come from one build: the builders refuse
to write beside a component from another checkpoint, option set or route, and the runtime refuses a
bundle whose components disagree. Bundles built before this check carry no identity and must be
rebuilt.

### From a CPU ONNX export

```bash
export PYTHONNOUSERSITE=1
tensorrt-edgellm-export "$CHECKPOINT" "$ONNX_DIR" --dtype float16

"$BUILD_DIR/bin/gr00t_policy_build" \
    --onnxDir "$ONNX_DIR" --engineDir "$ENGINE_DIR"
```

The export writes the same three component directories plus `tokenizer/`, each component beside
its sidecars -- the token-embedding table under `vl_backbone/`, the per-step constant tables and
`policy.json` under `action/` -- and the builder stages them into the layout above, so the runtime
cannot tell the two routes apart. `--gr00t-embodiment-id`, `--gr00t-max-vl-len` and
`--gr00t-max-image-len` on the export take the place of the builder flags, and `--components
<name>` exports one component. On `gr00t_policy_build`, `--component visual|vl_backbone|action`
compiles one component and `--maxBatchSize N` widens the batch axis. The `visual` tower goes
through the shared Qwen3-VL visual builder and needs the plugin library at build time as well.

## 2. Run the policy

```bash
"$BUILD_DIR/bin/gr00t_policy_inference" \
    --engineDir "$ENGINE_DIR" \
    --inputFile observation.json \
    --output action.json
```

```json
{
  "task": "pick up the black bowl and place it on the plate.",
  "state": [-0.39, 0.33, 0.85, 2.24, -1.48, -0.70, 0.012, -0.030],
  "cameras": {
    "image": "view0.png",
    "wrist_image": "view1.png"
  }
}
```

`cameras` is keyed by the names `policy.json` declares, so an unknown or repeated name is rejected,
and every camera is required; relative image paths resolve against the request file. `state` is in
the robot's native units, in the contract's joint-group order; LIBERO's gripper is two values, so its
state is 8 wide where its action is 7. `task` is lower-cased and stripped of punctuation before
tokenization, as the reference does.

`--image FILE` (repeatable, in contract order) with `--instruction` and `--state V,...` is the same
request from the command line. `--noCudaGraph` enqueues the four denoise steps individually instead
of replaying one captured graph, and `--noise FILE` replaces the seeded initial noise with a raw
float32 `[40, 132]` tensor.

`action.json` holds `actions`, the normalized zero-padded `[40, 132]` chunk the engine emits, and
`robot_actions`, the first 16 steps of the same chunk in robot units sliced to the embodiment's 7
dimensions.

Without `--inputFile` or `--image` the CLI synthesizes a VL context at a chosen shape (`--vlLen`,
`--imageLen`, `--batch`), which exercises the engines and the denoise loop but not the vision front
end.

## 3. Verify correctness

This is a numerical comparison against Isaac-GR00T's PyTorch chunk for one fixed observation, not
task success. Run both sides on the same observation and the same initial noise, then compare the
normalized chunks. `x_0` must come from the reference run: the two implementations draw from
different RNGs, so a shared seed does not give a shared trajectory.

```bash
# Same observation and x_0 both sides saw, so any difference is the engines.
"$BUILD_DIR/bin/gr00t_policy_inference" \
    --engineDir "$ENGINE_DIR" --inputFile observation.json \
    --noise ref/x0.bin --output actual.json

python experimental_models/scripts/compare_actions.py \
    --reference ref/actions.npy --actual actual.json --max-abs-ceiling 1e-2
```

The comparator reports cosine similarity and max absolute error over the normalized chunk and exits
non-zero outside its thresholds; `--json-field robot_actions` scores the robot-unit chunk instead.
`actions` needs the raised ceiling because the fp16 vision front end alone accounts for most of its
error; `robot_actions` clears the defaults. The chunk is dominated by its initial noise, so the
per-step velocity comparison in the design page is the figure that separates a right graph from a
wrong one; the measured values are recorded there.
