# InternVLA-N1-DualVLN

InternVLA-N1-DualVLN is a vision-language *navigation* model built from two systems that run at
different rates:

- **System 2** — a Qwen2.5-VL-7B planner. Standard VLM export; served by the core runtime.
- **System 1** — a flow-matching trajectory expert plus a memory block over recent frames.

They are joined by `z_latents`: the hidden states at the model's trajectory-query positions,
normalized and projected. That projection is folded into the LLM graph, so the engine emits
`z_latents` directly and nothing downstream has to carry projector weights or reproduce the
norm ordering.

The planner's *text* output is not what drives navigation. A checkpoint can produce fluent
replies and still be useless here; `z_latents` is the signal that matters.

## Export

One command produces all three components from the released checkpoint. No repackaging step is
needed — the exporter reads the InternVLA config directly.

```bash
tensorrt-edgellm-export /path/to/InternVLA-N1-DualVLN ./onnx
```

```
onnx/llm/model.onnx        + embedding.safetensors   System 2 decoder, emits z_latents
onnx/visual/model.onnx                               Qwen2.5-VL tower, unchanged
onnx/action/memory.onnx                              DINOv2 + temporal encoder + resampler
onnx/action/traj_dit.onnx                            one flow-matching denoising step
```

The System-1 component ships **two** graphs rather than one. The memory block runs once per
observation window while the trajectory expert runs once per denoising step, so a fused graph
would re-encode the frames on every step.

## Build

System 2 uses the standard executables:

```bash
llm_build    --onnxDir onnx/llm    --engineDir engines/llm \
             --maxBatchSize 1 --maxInputLen 3072 --maxKVCacheCapacity 4096
visual_build --onnxDir onnx/visual --engineDir engines \
             --minImageTokens 4 --maxImageTokens 4096 --maxImageTokensPerImage 1024
```

Size the visual engine for multi-image prompts. A navigation prompt carries roughly ten frames
and about 1764 image tokens; the single-image demo default of 512 cannot hold one.

System 1 builds with `trtexec`, since its two graphs do not share the single-`model.onnx`
layout `action_build` expects:

```bash
trtexec --onnx=onnx/action/traj_dit.onnx --saveEngine=engines/action/traj_dit_bf16.engine --bf16 \
        --minShapes=latents:64x32x3,timestep:64,z_latents:64x4x768  \
        --optShapes=latents:64x32x3,timestep:64,z_latents:64x36x768 \
        --maxShapes=latents:64x32x3,timestep:64,z_latents:64x64x768
trtexec --onnx=onnx/action/memory.onnx --saveEngine=engines/action/memory_bf16.engine --bf16 \
        --minShapes=images:1x3x224x224 --optShapes=images:2x3x224x224 --maxShapes=images:8x3x224x224
```

The trajectory batch is `2 * num_sample_trajs` because the sampler runs classifier-free
guidance: the conditioning is `[null, real]` and the latents are duplicated.

## Run

```bash
internvla_n1_system1_inference --engineDir engines/action \
    --conditioning cond.bin --noise noise.bin --output trajectory.bin \
    --numTrajs 32 --steps 10 --guidance 1.5
```

Tensors are read and written as raw float32 so a run can be reproduced and compared exactly.
The noise is supplied rather than drawn internally for the same reason.

## Notes

**FP16 on Jetson Thor needs a build flag.** TensorRT 10.13 miscompiles Myelin's horizontal
fusion of the gate/up projections at batch 1 on sm_110, and an engine built without the
workaround emits fluent gibberish. Export `__LUNOWUD=-peep:fc_h_fusion=off` before `llm_build`.
FP8 dodges the same bug because its Q/DQ nodes break the fusion pattern.

**System 1 stays BF16.** Quantizing it was measured: FP8 costs about six times the waypoint
deviation to save 0.7 % of deployed weights and 1.7 % of a planning step, because System 2
dominates both.

**Engine I/O is fp32/int64 even though the weights are BF16.** Passing bf16 tensors fails an
assertion rather than converting silently.

**The memory block expects normalized frames.** The reference divides by the ResNet statistics
before this block, and the block does not normalize again. Feeding raw pixels produces
plausible-looking but wrong tokens.

## Measured

Jetson Thor, idle GPU.

| | |
|---|---|
| trajectory loop, 10 steps x 32 samples | 46.4 ms (21.5 Hz) |
| same loop in Python | 61.8 ms |
| PyTorch reference | 175.4 ms |
| memory engine / trajectory engine | 109 MB / 72 MB |

Fidelity against the PyTorch reference, same weights: memory block cosine 0.99993, one
denoising step 0.99996, and the full C++ loop reproduces the Python loop at cosine 1.00000000
(max abs diff 4.5e-07).
