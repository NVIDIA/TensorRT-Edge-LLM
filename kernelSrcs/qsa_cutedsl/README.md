# QSA sparse-GQA prefill and decode kernels (CuTe DSL)

Qwen Sparse Attention (QSA, Qwen3.8-Flash-Next): the QSA indexer selects, per
query token, the top-512 KV blocks of 4 tokens, expands them to token ids and
appends the current partial block's tail — an int32 list of width
`indexer_budget + compress_ratio - 1 = 2051`, `-1`-padded, unsorted, distinct,
all `< token + 1`. The kernels here attend ONLY to the listed tokens; there is
no causal or positional mask in-kernel.

## Files

- `qsa_sparse_gqa.py` — `QSASparseGQAPrefill`: structural fork of
  `fmha_v2_cutedsl/fmha.py` (same swizzled SMEM atoms, m16n8k16 tiled MMA,
  ldmatrix double-buffered BMM loops and exp2 online softmax) with a sparse
  per-row cp.async gather replacing the affine KV traversal.
  `QSASparseGQADecode` subclasses it for single-token split-K decode: it
  reuses the per-chunk BMM + online softmax, replaces the index staging and
  K/V row gather with paged variants (`_load_chunk_meta`,
  `_gather_stage_paged`) and runs several independent warps per CTA that
  meet only in a cooperative merge epilogue.

## Prefill kernel shape

- Grid `(S, H_kv, B)` — one CTA per (query token, KV head).
- M tile = the token's GQA head group (`H_q / H_kv` rows, zero-padded to
  `m_block_size = 16`); head-padding rows are never loaded or stored.
- The index list is walked in `n_block_size = 16` chunks; one warp-wave
  cp.async-gathers one 256-element K/V row (32 lanes x 128 bits). Chunk
  indices are staged in a ping-pong SMEM buffer shared by the K gather, the
  V gather and the score mask.
- `-1` entries: zero-filled K/V rows AND `-inf` scores (NaN hardening).
- Padding query rows (`t >= context_lengths[b]`) and all-invalid rows store
  exact zeros (`row_sum == 0` guard).
- The softmax scale multiplies the FP32 scores at exp2 time — it is never
  pre-folded into the fp16/bf16 Q (matches the vLLM QSA numerics ordering).

Baked at compile time: `head_dim` (256), the `(Br, Bc, threads)` tuning and
`pipe_depth` (2, script default).
Runtime-dynamic: batch, seq, `H_q`, `H_kv` (group size <= 16), `topk`,
strides, softmax scale.

## Decode kernel shape

- Grid `(MAX_SPLITS, H_kv, B)`, fixed per captured CUDA graph — one CTA per
  (split, KV head, sequence); one query token per sequence
  (`q [B, 1, H_q, D]`).
- Every CTA derives the same `n_active = clamp(ceil(n_tiles / 4), 1,
  MAX_SPLITS)` from `context_lengths` (`n_tiles = ceil(min(topk, ctx) /
  n_block_size)`; at least 4 chunks per split keeps the `pipe_depth`-stage
  pipeline filled) and its contiguous tile slice. Inactive splits do no compute but
  still arrive at the merge counter.
- CTA = `num_threads / 32` warps (4 by default). Warp `w` owns every
  `num_warps`-th chunk of the CTA's slice and runs the single-warp
  (16 x `n_block_size`) MMA pipeline over them with its own cp.async K/V
  stages, index + page-id prefetch (two chunks ahead of the gather that
  consumes it, branch-free clamped loads) and online-softmax state; the warps
  never synchronize in the mainloop. Decode is gather-latency bound — at B=1
  the grid is 16 CTAs on Thor's 20 SMs — so the extra warps exist to keep
  more row gathers in flight per SM, not for MMA throughput.
- Epilogue: each warp stages its unnormalized fp32 accumulator and per-row
  `(max, sum)` in smem, aliasing its own (idle) K stages for the low half of
  the columns and V stages for the high half; then all threads merge the
  warps row-wise (warp `w` owns rows `w, w + num_warps, ...`, a lane owns
  4 consecutive columns per half row, 128-bit accesses).
- K/V are gathered from the Edge-LLM paged pool
  `[2 * num_pages, 128, H_kv, pool_head_dim]` through the page table
  `[B, 2, max_pages_per_seq]` (V page ids pre-offset by `+num_pages`). Only
  columns `[0, head_dim)` of each pool row are read; the tail
  `[head_dim, pool_head_dim)` is QSA indexer state.
- `n_active == 1` stores the normalized O straight from that merge. Otherwise
  each active CTA writes the merged unnormalized fp32 partial plus per-row
  `(max, sum)` stats, fences and bumps a per-`(b, kv_head)` counter; the last
  arriver (tested against the static `MAX_SPLITS` denominator) merges the
  splits with the FA split-K rescale using the same row-wise mapping
  (branch-free over `MAX_SPLITS`: inactive slots re-read the last active
  split with weight 0), stores O and release-stores the counter back to 0. Counters must
  be zero on entry: the self-reset covers steady state, and the plugin zeroes
  them in-enqueue (indexer decode B3) so the first launch on a fresh
  workspace is safe as well.
- `context_lengths[b]` is the TOTAL length including the token being decoded.

Baked at compile time: `head_dim` (256), the `(Br, Bc, threads)` tuning
(16, 16, 128), `pipe_depth` (1 stage per warp) and `MAX_SPLITS` (8) — the
decode defaults live in `QSA_DECODE_DEFAULT_THREADS` /
`QSA_DECODE_DEFAULT_PIPE_DEPTH` and are pinned to the registry by
`test_build_cutedsl_helpers.py`. Runtime-dynamic: batch, `H_q`, `H_kv`,
`topk`, `pool_head_dim`, page geometry, softmax scale.

## Variants (`build_cutedsl.py` group `qsa`)

| name | dtype | mode | SMs |
|---|---|---|---|
| `qsa_sparse_d256_fp16` | fp16 | prefill | 100, 101, 110 |
| `qsa_sparse_d256_bf16` | bf16 | prefill | 100, 101, 110 |
| `qsa_sparse_decode_d256_fp16` | fp16 | decode (`--decode --max_splits 8`) | 100, 101, 110 |
| `qsa_sparse_decode_d256_bf16` | bf16 | decode (`--decode --max_splits 8`) | 100, 101, 110 |

All four bake `--head_dim 256 --m_block_size 16 --n_block_size 16`; the
prefill variants use `--num_threads 32` (one warp per query token), the
decode variants `--num_threads 128 --pipe_depth 1` (4 gather warps per
CTA).

## Standalone test / benchmark (needs a GPU)

Prefill:

```bash
python kernelSrcs/qsa_cutedsl/qsa_sparse_gqa.py \
    --batch_size 2 --seqlen 384 --num_head 24 --kv_group_size 12 \
    --head_dim 256 --topk 2051 --ragged
```

The in-file FP32 CuPy oracle checks the gather-softmax result against
NaN-poisoned padding and a sentinel-poisoned output buffer (padding rows must
come back as exact zeros).

Decode:

```bash
python kernelSrcs/qsa_cutedsl/qsa_sparse_gqa.py --decode \
    --batch_size 4 --seqlen 8192 --decode_steps 4 --num_head 24 \
    --kv_group_size 12 --head_dim 256 --topk 2051 --ragged
```

`--seqlen` is the TOTAL context of sequence 0 at the first decode step (with
`--ragged` the other sequences get random shorter contexts); every further
step appends one token per sequence. The pool is fully NaN-poisoned behind a
permuted, non-identity page table, with `--pool_head_dim` (default 384) wide
rows whose `[head_dim, pool_head_dim)` tails stay NaN, so any read of the
indexer-state tail or of an unmapped page fails the check. Every step is
compared against the FP32 CuPy oracle and launched twice, requiring
bitwise-identical outputs (validates the counter self-reset).

## AOT build

```bash
python kernelSrcs/build_cutedsl.py --kernels qsa --gpu_arch sm_110 --arch aarch64
```

Consumed by `cpp/kernels/qsaAttention/cuteDslQsaSparseRunner.{h,cpp}`
(`CuteDslQsaSparsePrefillRunner`, `CuteDslQsaSparseDecodeRunner`) under
`CUTE_DSL_QSA_ENABLED` (cmake `-DENABLE_CUTE_DSL="fmha;qsa"`).
