# GR00T N1.7 (experimental runtime)

GR00T N1.7 has two build routes that write the same engine bundle: the ONNX-less direct builder
under `tensorrt_edgellm/builder/models/gr00t_n1d7` (`tensorrt-edgellm-build`, on the target), and a CPU
ONNX export under `tensorrt_edgellm/models/gr00t_n1d7` (`tensorrt-edgellm-export`) compiled by the
`gr00t_policy_build` CLI here. This directory keeps the experimental runtime and that builder: the
vision front end, the policy runner, the device-side Euler update and the two CLIs.

## Layout

```text
gr00t_n1d7/
  cpp/action/     # policy runner, robot-unit policy, Euler kernel
  cpp/builder/    # component builder for the ONNX route
  cpp/multimodal/ # vision front end: eval transform, chat template, embedding assembly
  cpp/runtime/    # Gr00tRuntime: one observation in, one action chunk out
  examples/       # gr00t_policy_build, gr00t_policy_inference
```

## Build

```bash
export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc

cmake -S <repo> -B "$BUILD_DIR" \
    -DTRT_PACKAGE_DIR=/usr -DCUDA_DIR=/usr/local/cuda \
    -DBUILD_EXPERIMENTAL_MODELS=ON -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUTE_DSL=ALL -DCUTE_DSL_ARTIFACT_TAG=sm_110
make -C "$BUILD_DIR" gr00t_policy_build gr00t_policy_inference NvInfer_edgellm_plugin -j"$(nproc)"
export EDGELLM_PLUGIN_PATH="$BUILD_DIR/lib/libNvInfer_edgellm_plugin.so"
```

## Quickstart

```bash
# 0. The Hub checkpoint has no tokenizer; stage the backbone's under $CHECKPOINT/tokenizer first
#    (see docs/source/user_guide/examples/vla/gr00t_n1d7.md).
# 1. Compile the three component engines from the checkpoint.
.venv/bin/tensorrt-edgellm-build \
    --model-dir "$CHECKPOINT" --engine-dir "$ENGINE_DIR" --components all \
    --plugin-path "$EDGELLM_PLUGIN_PATH" \
    --max-input-len 1024 --max-batch-size 1

#    Or export on CPU and build from ONNX; the bundle is the same.
tensorrt-edgellm-export "$CHECKPOINT" "$ONNX_DIR" --dtype float16
"$BUILD_DIR/bin/gr00t_policy_build" \
    --onnxDir "$ONNX_DIR" --engineDir "$ENGINE_DIR"

# 2. Run the policy.
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
"$BUILD_DIR/bin/gr00t_policy_inference" \
    --engineDir "$ENGINE_DIR" --inputFile observation.json --output action.json
```

`observation.json` carries `task`, `state` in robot units and `cameras` keyed by the contract's
names (`image`, `wrist_image`); `--image`/`--instruction`/`--state` spell the same request on the
command line. `experimental_models/scripts/compare_actions.py` scores `action.json` against a reference chunk.

The `visual` engine carries a `trt_edgellm::ViTAttentionPlugin` node, so it needs the Edge-LLM plugin
library at run time; the other two are plugin-free.

See `docs/source/developer_guide/models/gr00t_n1d7.md` for the full contract.
