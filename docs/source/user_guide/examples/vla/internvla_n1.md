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

**NVFP4 needs a second build flag on TRT 10.13.** The CASK epilogue fusion miscompiles NVFP4
at batch 1, and the resulting engine is both wrong and *faster* -- 62.3 ms against 72.8 ms for
the correct one, because a miscompiled kernel does less work. A number that good from an
unpatched build is the symptom, not a win. Add `-cask_fusion:max_num_epilogues=1` to
`__LUNOWUD`, or build with `--maxBatchSize 2`, which sidesteps it at no cost.

**System 1 stays BF16.** Quantizing it was measured: FP8 costs about six times the waypoint
deviation to save 0.7 % of deployed weights and 1.7 % of a planning step, because System 2
dominates both.

**Engine I/O is fp32/int64 even though the weights are BF16.** Passing bf16 tensors fails an
assertion rather than converting silently.

**The memory block expects normalized frames.** The reference divides by the ResNet statistics
before this block, and the block does not normalize again. Feeding raw pixels produces
plausible-looking but wrong tokens.

## The two systems run at different rates

System 2 plans in roughly 646 ms; System 1 produces a trajectory in 46 ms. The head is meant
to keep running on the newest plan available rather than waiting for a fresh one, so the
runtime puts the planner on its own thread (`InternVLAN1DualSystemDriver`) and hands plans over
through shared state that publishes atomically.

The planner is injected, not owned: what a plan *is* -- which frames, which prompt, which
engine -- belongs to the deployment. What the runtime guarantees is that a slow planner cannot
stall the trajectory loop.

Two consequences worth stating plainly:

- **Running on a stale plan is normal, not a failure.** `stalenessAt()` reports how many
  observations old the current plan is, so a caller can bound it.
- **Give System 1 a high-priority stream.** It produces the control output, so it is the
  workload that must not be starved; System 2 is allowed to take longer.
  `InternVLAN1System1Runner::makeControlStream()` creates one at the device's greatest
  priority. Measured against a competing loop: 94.0 ms per trajectory at equal priority against
  73.3 ms with it, and the competing loop was unaffected within noise. Alone it is 48.2 ms, so
  priority recovers about half of what contention costs -- CUDA preempts between kernels, not
  inside one, so the rest is not recoverable this way.
- **The gain is latency hiding, not parallel throughput.** With System 2 running, the
  trajectory loop drops from 20.7 Hz to 8.2 Hz -- the two contend for the GPU rather than
  overlapping. Asynchrony is still what you want: without it the head stalls completely for the
  ~646 ms System 2 takes, and 8.2 Hz throughout beats zero followed by a burst. The separate
  context pool is a correctness property -- the two cannot corrupt each other's scratch -- not
  a speed one.

## Measured

Jetson Thor, idle GPU, batch 1, measured with `llm_bench`.

### System 2

| Variant | prefill (1024 tokens) | decode (pastKV 1024) | weights |
|---|---|---|---|
| PyTorch fp16 | 328.93 ms | 99.35 ms | ~15 GB |
| TensorRT FP16 | 150.80 ± 1.78 ms | 63.97 ± 5.79 ms | 14.15 GB |
| TensorRT FP8 | 90.17 ± 0.48 ms | 33.03 ± 0.30 ms | 7.62 GB |
| TensorRT NVFP4 | 72.83 ± 0.42 ms | 23.33 ± 0.90 ms | 4.77 GB |

Against PyTorch that is 2.2x/1.6x for FP16, 3.6x/3.0x for FP8 and 4.5x/4.3x for NVFP4. The
PyTorch row runs the same decoder over the same input length and past-KV length as `llm_bench`,
so the rows are comparable; it is not the model's end-to-end agent latency.

FP8 is the recommended scheme: 1.86x smaller than FP16 and roughly 1.7x/1.9x faster, with the
navigation bridge measured at 0.9919 in the source recipe. NVFP4 is smaller and faster still
but its bridge falls to 0.931, below the 0.99 gate, so it is not recommended for navigation
despite the numbers above.

The FP16 engine carries one extra output — the bridge — that the FP8 and NVFP4 engines here do
not, since those were quantized from an already-repackaged checkpoint. The difference is one
tensor and does not move these figures, but the rows are not byte-identical graphs.

### System 1

| | |
|---|---|
| trajectory loop, 10 steps x 32 samples | 48.3 ms (20.7 Hz) |
| the same loop with System 2 running | 121.6 ms (8.2 Hz) |
| the same loop in Python | 61.8 ms |
| PyTorch reference | 175.4 ms |
| memory engine / trajectory engine | 109 MB / 72 MB |

With a planner 14x slower than the trajectory loop running concurrently, the loop's worst
single tick grew by under 8 % and replan requests coalesced 15 into 5 — a request arriving
while one is in flight replaces the pending one rather than queueing behind it.

Fidelity against the PyTorch reference, same weights: memory block cosine 0.99993, one
denoising step 0.99996, and the full C++ loop reproduces the Python loop at cosine 1.00000000
(max abs diff 4.5e-07).

