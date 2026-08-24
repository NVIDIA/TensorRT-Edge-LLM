# InternVLA-N1-DualVLN (experimental runtime)

Model definitions and ONNX export live in the main Python package under
`tensorrt_edgellm.models.internvla_n1` (exported through the unified
`tensorrt-edgellm-export`). This directory keeps the experimental runtime: the C++
System-1 trajectory runner, the asynchronous dual-system state, and their CLIs.

The model is a vision-language *navigation* model built from two systems that run at
different rates: a **Qwen2.5-VL-7B planner** (System 2, standard VLM export) and a
**flow-matching trajectory expert** plus memory block over recent frames (System 1). They are
joined by `z_latents` — the hidden states at four learned trajectory-query embeddings,
normalized and projected. The queries travel as real tokens: the export writes them into the
embedding table's trailing padding rows and registers `<|latent_q0..3|>`, and folds the final
norm and `cond_projector` into the LLM graph, so the engine emits `z_latents` directly at
`latent_dim` (recorded as `output_hidden_size` in the engine's `config.json`).

The planner's *text* output is not what drives navigation, so validation compares `z_latents`
and trajectories against the reference, not text. The supported input is the released
`InternVLA-N1-DualVLN` checkpoint.

## Layout

```text
internvla_n1/
  cpp/
    action/    # System-1 runner: memory + traj_dit engines, flow-matching sampler, CFG
    runtime/   # dual-system state + driver: atomic plan handoff, background planner thread
  examples/    # one-shot CLIs (…_inference) and resident servers (…_server)
```

System 1 ships **two** graphs rather than one: the memory block runs once per observation
window while the trajectory expert runs once per denoising step, so a fused graph would
re-encode the frames on every step. Neither fits `action_build`'s one-graph shape, so both are
built with `trtexec`.

## Quickstart

Binaries come from the standard experimental-model build (`-DBUILD_EXPERIMENTAL_MODELS=ON`);
`$BUILD_DIR` points at that build tree and `$EX` at its `experimental_models/internvla_n1/examples`.

```bash
# 1. Export ONNX (the unified exporter detects InternVLA-N1 automatically). Produces
#    $ONNX_DIR/llm (+ embedding.safetensors carrying the latent queries), /visual,
#    and /action (memory.onnx + traj_dit.onnx).
tensorrt-edgellm-export /path/to/InternVLA-N1-DualVLN "$ONNX_DIR"

# 2. Build System 2 with the shared builders.
export EDGELLM_PLUGIN_PATH="$BUILD_DIR/libNvInfer_edgellm_plugin.so"
"$BUILD_DIR/examples/llm/llm_build" --onnxDir "$ONNX_DIR/llm" --engineDir "$ENGINE_DIR/llm" \
    --maxBatchSize 1 --maxInputLen 3072 --maxKVCacheCapacity 4096
"$BUILD_DIR/examples/multimodal/visual_build" --onnxDir "$ONNX_DIR/visual" \
    --engineDir "$ENGINE_DIR" --minImageTokens 4 --maxImageTokens 4096 --maxImageTokensPerImage 1024

# 3. Build System 1 (two graphs, trtexec).
trtexec --onnx="$ONNX_DIR/action/traj_dit.onnx" --saveEngine="$ENGINE_DIR/action/traj_dit.engine" \
    --bf16 --minShapes=latents:64x32x3,timestep:64,z_latents:64x4x768 \
    --optShapes=latents:64x32x3,timestep:64,z_latents:64x36x768 \
    --maxShapes=latents:64x32x3,timestep:64,z_latents:64x64x768
trtexec --onnx="$ONNX_DIR/action/memory.onnx" --saveEngine="$ENGINE_DIR/action/memory.engine" \
    --bf16 --minShapes=images:1x3x224x224 --optShapes=images:2x3x224x224 --maxShapes=images:8x3x224x224

# 4. Run and time both systems asynchronously in one process: System 2 plans on a
#    background thread while System 1 keeps sampling from the newest plan. Reports
#    the first-plan latency (System 2), the mean System-1 tick, and how many plans
#    landed -- which together are the numbers that matter for deployment.
"$EX/internvla_n1_dual_system_inference" --llmEngineDir "$ENGINE_DIR/llm" \
    --actionEngineDir "$ENGINE_DIR/action" --frames frames.bin --noise noise.bin \
    --ticks 40 --cadence 4 --output traj.bin
```

`--cadence` is what makes the measurement realistic. Replanning every 4 ticks puts the mean
System-1 tick at 55.4 ms against 50.2 ms with replanning effectively off -- the gap is System 2
competing for the GPU, which is what a deployed agent actually pays. Measured on Jetson Thor
with a 7B planner, System 1 in BF16 and 32 trajectories over 10 denoising steps:

| System 2 | First plan | Control rate | Engine |
|---|---|---|---|
| FP16 | 144 ms | 66.8 ms (15.0 Hz) | 13.2 GB |
| FP8 | 118 ms | 61.3 ms (16.3 Hz) | 7.1 GB |
| NVFP4 | 100 ms | 55.4 ms (18.0 Hz) | 4.5 GB |

Quantizing System 2 raises the control rate even though System 1 is untouched, because the
planner stops crowding it: the trajectory head alone runs at 48 ms regardless of which
planner it shares the GPU with.

Guidance defaults to 1.0 — the value every `generate_traj` call site in InternNav deploys
with. Frames must already be normalized with the ResNet statistics; the memory block does not
normalize again.

## Driving the systems from an agent

Step 5 runs on canned tensors. An agent stepping a simulator or a robot uses
`internvla_n1_dual_system_server` instead: both engines resident in one process, one JSON
request per stdin line. A benchmark run is tens of thousands of steps, so spawning a CLI per
step would reload a 14 GB engine each time. It takes three request kinds — `text` for a
System-2 decision, `replan` to wake the planner and return immediately, and `trajectory` for
System 1 against whatever plan is current, which also reports how many observations old that
plan is.

One process is not incidental: CUDA orders streams within a context, so System 1's priority
stream only outranks the planner when the two share one. Across processes the GPU time-slices
and the priority is invisible — 73.3 ms per trajectory in-process against 121 ms split, under a
competing load.

Tensors travel as raw float32 behind the request header, not as JSON arrays: a 2x3x224x224
frame window is 301,056 values, and parsing that as text cost 545 ms per step — more than every
engine in the pipeline put together.

See `docs/source/user_guide/examples/vla/internvla_n1.md` for the architecture, measured
latency/fidelity, and the platform build notes.
