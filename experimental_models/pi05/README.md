# pi0.5 (experimental runtime)

pi0.5 model definitions and ONNX export live in the main Python package under
`tensorrt_edgellm.models.pi05` (exported through the unified `tensorrt-edgellm-export`). This
directory keeps the separate experimental runtime: C++ component builder, policy runner and
inference CLI.

The supported input is an already-converted **PyTorch** pi0.5 checkpoint (`model.safetensors` +
`config.json`); the LeRobot releases are already in that form. This runtime supports openpi's
`pi05_libero`, `pi05_droid` and `pi05_aloha` contracts, in FP16.

## Layout

```text
pi05/
  cpp/        # component builder, policy runner, policy pre/post-processing
  examples/   # pi05_policy_build and pi05_policy_inference
  scripts/    # compare_pi05_actions.py, the numerical comparator
```

## Quickstart

The binaries come from the standard Edge-LLM experimental-model build (configure with
`-DBUILD_EXPERIMENTAL_MODELS=ON`); this assumes they and the plugin library are already built.

```bash
export CHECKPOINT=lerobot/pi05_libero_base ONNX_DIR=$HOME/pi05_onnx
export ENGINE_DIR=$HOME/pi05_engines BUILD_DIR=/path/to/tensorrt-edge-llm/build
export EDGELLM_PLUGIN_PATH="$BUILD_DIR/libNvInfer_edgellm_plugin.so"

# 1. Export ONNX + component contracts (x86 host, CPU-only). --pi05-policy-config names the
#    openpi contract, and is required for every configuration but pi05_libero.
PYTHONNOUSERSITE=1 tensorrt-edgellm-export "$CHECKPOINT" "$ONNX_DIR" --dtype float16 \
    --pi05-policy-config pi05_libero

# 2. Stage the configuration's normalization statistics. The checkpoint carries the
#    processor schemas but not their state, which stays in openpi's own checkpoint
#    assets. They have to be here before step 3, which copies them into the bundle.
mkdir -p "$ONNX_DIR/assets"
curl --fail --location --output "$ONNX_DIR/assets/norm_stats.json" \
  https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json

# 3. Build every component engine and stage the runtime sidecars.
"$BUILD_DIR/experimental_models/pi05/examples/pi05_policy_build" \
    --onnxDir "$ONNX_DIR" --engineDir "$ENGINE_DIR"

# 4. Run policy inference. See the example doc for the observation.json format.
"$BUILD_DIR/experimental_models/pi05/examples/pi05_policy_inference" \
    --engineDir "$ENGINE_DIR" --inputFile observation.json --output action.json
```

The export writes four components: `visual/`, `prefix/`, `action/` and `cond/`. `cond/` precomputes
the AdaRMS modulation schedule instead of rebuilding it inside the per-step action graph, and the
runtime re-evaluates it only when the denoise-step count or the batch changes. It is optional:
`--no-pi05-hoist-adarms-cond` leaves it out.

[pi0.5 example](../../docs/source/user_guide/examples/vla/pi05.md) has the statistics download, the
request format and the correctness check; [pi0.5 Design](../../docs/source/developer_guide/models/pi05.md)
has the component contracts.
