# InternVLA-N1 (experimental runtime)

InternVLA-N1-DualVLN model definitions and ONNX export live in the main Python package under `tensorrt_edgellm.models.internvla_n1` (exported through the unified `tensorrt-edgellm-export`). This directory keeps the separate experimental runtime: the C++ System-1 trajectory runtime, the asynchronous dual-system state, and their inference CLIs.

The model is a vision-language *navigation* model built from two systems that run at different rates:

- **System 2** — a Qwen2.5-VL-7B planner. Standard VLM export, served by the core LLM runtime.
- **System 1** — a flow-matching trajectory expert (cross-attention DiT) plus a memory block (DINOv2 + temporal encoder + resampler) over recent frames.

They are joined by `z_latents`: the hidden states at four learned trajectory-query embeddings, normalized and projected. The queries travel as real tokens — the export writes them into the embedding table's trailing padding rows and registers `<|latent_q0..3|>` as special tokens (IDs recorded in `config.json` under `latent_query_token_ids`), so a prompt ending with those tokens puts them into the sequence through the runtime's ordinary embedding lookup. The final norm and `cond_projector` are folded into the LLM graph, so the engine emits `z_latents` directly.

The planner's *text* output is not what drives navigation — a checkpoint can produce fluent replies and still be useless here — so validation compares `z_latents` and trajectories against the reference, not text.

The supported input is the released `InternVLA-N1-DualVLN` checkpoint.

## Layout

```text
internvla_n1/
  cpp/
    action/    # System-1 runner: memory + traj_dit engines, flow-matching sampler, CFG
    runtime/   # dual-system state + driver: atomic plan handoff, background planner thread
  examples/    # internvla_n1_system1_inference, internvla_n1_dual_system_inference,
               # internvla_n1_dump_z_latents
```

System 1 ships **two** graphs rather than one: the memory block runs once per observation window while the trajectory expert runs once per denoising step, so a fused graph would re-encode the frames on every step. Neither fits `action_build`'s one-graph shape, so both are built with `trtexec`.

## Quickstart

The binaries are produced by the standard Edge-LLM experimental-model build (configure with `-DBUILD_EXPERIMENTAL_MODELS=ON`); this Quickstart assumes they, the shared `llm_build`/`visual_build`, and the plugin library are already built. `$BUILD_DIR` points at that build tree.

```bash
# 1. Export ONNX from the released checkpoint (the unified exporter detects
#    InternVLA-N1 automatically). Produces $ONNX_DIR/llm (+ embedding.safetensors
#    with the latent queries in its trailing rows, tokenizer with the
#    <|latent_q*|> tokens), $ONNX_DIR/visual, and $ONNX_DIR/action
#    (memory.onnx + traj_dit.onnx).
tensorrt-edgellm-export /path/to/InternVLA-N1-DualVLN "$ONNX_DIR"

# 2. Build System 2 with the shared builders.
export EDGELLM_PLUGIN_PATH="$BUILD_DIR/libNvInfer_edgellm_plugin.so"
"$BUILD_DIR/examples/llm/llm_build" \
    --onnxDir "$ONNX_DIR/llm" --engineDir "$ENGINE_DIR/llm" \
    --maxBatchSize 1 --maxInputLen 3072 --maxKVCacheCapacity 4096
"$BUILD_DIR/examples/multimodal/visual_build" \
    --onnxDir "$ONNX_DIR/visual" --engineDir "$ENGINE_DIR" \
    --minImageTokens 4 --maxImageTokens 4096 --maxImageTokensPerImage 1024

# 3. Build System 1 (two graphs, trtexec).
trtexec --onnx="$ONNX_DIR/action/traj_dit.onnx" \
        --saveEngine="$ENGINE_DIR/action/traj_dit.engine" --bf16 \
        --minShapes=latents:64x32x3,timestep:64,z_latents:64x4x768 \
        --optShapes=latents:64x32x3,timestep:64,z_latents:64x36x768 \
        --maxShapes=latents:64x32x3,timestep:64,z_latents:64x64x768
trtexec --onnx="$ONNX_DIR/action/memory.onnx" \
        --saveEngine="$ENGINE_DIR/action/memory.engine" --bf16 \
        --minShapes=images:1x3x224x224 --optShapes=images:2x3x224x224 \
        --maxShapes=images:8x3x224x224

# 4a. Trajectory head alone (conditioning from a file, reproducible).
"$BUILD_DIR/experimental_models/internvla_n1/examples/internvla_n1_system1_inference" \
    --engineDir "$ENGINE_DIR/action" \
    --conditioning cond.bin --noise noise.bin --output trajectory.bin \
    --numTrajs 32 --steps 10 --guidance 1.0

# 4b. Both systems asynchronously in one process: System 2 plans on a background
#     thread while System 1 keeps sampling from the newest plan.
"$BUILD_DIR/experimental_models/internvla_n1/examples/internvla_n1_dual_system_inference" \
    --llmEngineDir "$ENGINE_DIR/llm" --actionEngineDir "$ENGINE_DIR/action" \
    --frames frames.bin --noise noise.bin --ticks 40 --cadence 4 --output traj.bin
```

Guidance defaults to 1.0 — the value every `generate_traj` call site in InternNav deploys with. Tensors are raw float32 files so a run can be reproduced and compared exactly; frames must already be normalized with the ResNet statistics (the memory block does not normalize again).

Two conventions consumers must know:

- The engine emits `z_latents` at `latent_dim` rather than model width, so the export writes `output_hidden_size` into the engine's `config.json` and the runtime sizes its hidden-states buffer from that. `getBaseModelHiddenStates` then reports the real width. Without the key the runtime would fall back to model width and copy past the engine's output; `internvla_n1_dump_z_latents` is the quickest way to confirm a fresh export (a shape of `[1, seq, 768]` and a cosine near 1.0 against the reference).
- One process for the dual-system loop is not incidental: CUDA orders streams within a context, so System 1's priority stream only outranks the planner when the two share one.

See `docs/source/user_guide/examples/vla/internvla_n1.md` for the architecture, measured latency/fidelity, and the platform build notes.
