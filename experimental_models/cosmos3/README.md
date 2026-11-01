# Cosmos3-Omni (experimental runtime)

Cosmos3 model definitions and ONNX export live in the main Python package under `tensorrt_edgellm.models.cosmos3` (exported through the unified `tensorrt-edgellm-export`). This directory keeps the separate experimental runtime: C++ component builder, policy inference CLI, and RoboLab HTTP wrapper.

The supported input is an already-converted HF/diffusers-format Cosmos3 checkpoint with `transformer/`, `vae/`, `scheduler/`, and `text_tokenizer/`. DCP conversion is not part of the normal workflow.

## Layout

```text
cosmos3/
  cpp/        # separate C++ runtime, component builder, scheduler, runners
  examples/   # cosmos3_policy_build and cosmos3_policy_inference
  robolab/    # optional HTTP server/client integration
```

## Quickstart

The `cosmos3_policy_build` and `cosmos3_policy_inference` binaries are produced
by the standard Edge-LLM experimental-model build (configure with
`-DBUILD_EXPERIMENTAL_MODELS=ON`); this Quickstart assumes they and the plugin
library are already built. `$BUILD_DIR` below points at that build tree.

```bash
# 1. Export ONNX + component contracts from an HF/diffusers checkpoint
#    (the unified exporter detects Cosmos3 checkpoints automatically).
#    The policy variables default to the canonical request
#    (--action-chunk-size 16 --num-frames 17 --fps 5); override any of them
#    to reshape the action chunk / rollout the engines are built for.
export PYTHONNOUSERSITE=1
tensorrt-edgellm-export "$HF_CHECKPOINT" "$ONNX_DIR" --dtype float16 --task policy

# 2. Build all component engines in one command. cosmos3_policy_build reads the
#    export root, builds und_prefill/gen/vae_encoder into $ENGINE_DIR/<component>,
#    and stages the tokenizer + token-embedding table into $ENGINE_DIR so the
#    engine directory is self-contained (like llm_build). Pass --component to
#    build a single one.
"$BUILD_DIR/experimental_models/cosmos3/examples/cosmos3_policy_build" \
    --onnx-dir "$ONNX_DIR" --engine-dir "$ENGINE_DIR"

# 3. Run policy inference.
export EDGELLM_PLUGIN_PATH="$BUILD_DIR/libNvInfer_edgellm_plugin.so"
"$BUILD_DIR/experimental_models/cosmos3/examples/cosmos3_policy_inference" \
    --image "$IMAGE" --prompt "$PROMPT" \
    --engine-dir "$ENGINE_DIR" --output action.json
```

See `docs/source/developer_guide/models/cosmos3.md` for the full contract.
