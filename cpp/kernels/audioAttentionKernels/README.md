# Gemma 4 Audio Encoder Attention Kernel

Fused CUDA kernel for the Gemma 4 audio-encoder chunked local attention body —
the region *after* Q/K/V projection and *before* the output projection. Entry
point: `trt_edgellm::kernel::gemma4AudioAttentionForward` (`gemma4AudioAttention.h`).
It takes `rt::Tensor` inputs — the shapes `B/S/H/D` (from `qRaw`) and `P` (from
`relKey`) and the element type (fp16/bf16/fp32) are read from the tensors, so only
the attention-window config (`chunkSize`, `leftHorizon`, `contextSize`, `logitCap`)
is passed via `Gemma4AudioAttentionParams`.

## What it computes

Per-dim learned Q scaling (`softplus(gamma)`) + fixed K scaling, overlapping
24-slot K/V context gather, content scores, query-dependent relative-position
scores with the HF relative shift, tanh soft-cap (50), local-causal + padding
mask, fp32 softmax over the 24-slot context, value mix, and crop back to seqLen.
One CUDA block handles one `(b, h, n)` chunk; the three per-chunk GEMMs and the
epilogue are fully fused.

Specialized to the Gemma 4 audio config (`chunkSize=12`, `leftHorizon=12`,
`contextSize=24`, `relPosLen=13`, `headDim=128`). Supports **fp16 / bf16 / fp32**
(internal scores/softmax are always fp32). Q/K/V/P_h are staged in shared memory
to keep occupancy off the shared-memory cap; shared loads are 2-wide vectorized
(`__half2`/`__nv_bfloat162`/`float2`); the content dot is skipped for masked
slots; shape constants are `constexpr` so the rel_shift div/mod fold to mul-shift.

The **fp32 path is the HF-faithful mode**: HF's Gemma 4 audio attention core is
itself fp32 (`q/k/v.float()`), so the float kernel keeps full precision throughout
and reproduces HF to round-off. fp16/bf16 trade precision for speed (they stage
the scaled Q/K in low precision).

## Status (RTX 3090, sm_86)

- **Accuracy** vs the fp32 reference, worst-case over a B/S/mask sweep:
  - **fp32: ~3.5e-6** (round-off — numerically equivalent to HF)
  - fp16 ≤ 2.9e-3, bf16 ≤ 2.0e-2
- **Performance** (CUDA-graph replay): ~5–8× (fp32) / ~3–5.7× (fp16/bf16) faster
  than an eager PyTorch implementation of the same attention body; end-to-end
  (with the external Q/K/V projections) ~2× for a bf16 model.
- ncu: "compute and memory well-balanced"; bound by the L1/shared-memory pipe and
  a shared-memory-limited ~33% occupancy (DRAM only ~15%, fp16 path).

Unit tests:
`unittests/cpp/kernels/audioAttentionKernels/gemma4AudioAttentionTests.cpp`
(kernel vs CPU fp32 reference, **fp32 / fp16 / bf16** shape/mask sweep).

## Correctness invariants (any change must preserve)

- fp32 softmax over the M context slots (inputs may be fp16/bf16).
- Exact relative shift index map: `aSrc = (a*M+m)/(M+1)`, `tSrc = (a*M+m)%(M+1)`.
- Exact mask: `local & jIn & iIn & valid[b,j]`, horizon `L = attention_context_left-1`.
- Soft-cap before mask before softmax; masked slots contribute 0 to the value
  mix; a fully-masked live row (e.g. a padding query) softmaxes to uniform, as
  in the HF reference (masked-but-live slots use the finite fill `-1e9`, padded
  tile slots use `-inf`).

## Future improvements

Bound is the shared/LSU pipe and the 33% occupancy cap, in this order of value:

1. **Raise occupancy past the shared cap** — stop staging V in shared (read it
   from global in the value mix; DRAM is only ~15%) and/or move `relKey` to
   constant memory. Dropping V (~6 KB) → ~6 blocks/SM (~50% occupancy).
2. **Register-tile the dot products** — reuse a loaded Q/K element across output
   columns in registers to cut the shared-pipe traffic that currently binds.
3. **Multi-chunk per CTA** — adjacent chunks' K/V contexts overlap by `L=12`; a
   sliding shared window cuts K/V staging toward the ~2× theoretical minimum.
4. **All-heads per CTA** — mask, rel_shift index map, and integer index math are
   head-independent; compute once and amortize across the 8 heads.
5. **Fuse Q/K scaling into the upstream projection epilogue**; INT8/FP8 GEMMs
   (keep softmax fp32).

**Dead end (measured):** tensor cores were ~1.5–1.9× *slower* — the per-chunk
GEMMs are tiny (12×24, 12×13, 12×128) so a 16×16×16 MMA tile is mostly padding,
and the op is memory/latency-bound, not flop-bound. Don't revisit unless the
kernel first becomes flop-bound (it is far from it).

## Generality note

The kernel supports fp16/bf16/fp32 but is hard-specialized to the Gemma 4 audio
shape (`constexpr` C/L/M/P/D). To support other configs, generalize the
`constexpr` block shape (and the shared-memory layout that depends on it) or fall
back to a reference path.
