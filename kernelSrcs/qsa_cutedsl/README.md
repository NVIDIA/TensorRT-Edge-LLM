# QSA sparse-GQA prefill and decode kernels (CuTe DSL)

Qwen Sparse Attention (QSA, Qwen3.8-Flash-Next): the QSA indexer selects, per
query token, the top-512 KV blocks of 4 tokens, expands them to token ids and
appends the current partial block's tail — an int32 list of width
`indexer_budget + compress_ratio - 1 = 2051`, right-padded with `-1`. Valid
entries are unsorted, distinct absolute KV token ids, all
`< past_lengths[b] + t + 1` for local query row `t`. Initial prefill has zero
past. The kernels here attend ONLY to the listed tokens; the indexer enforces
causality, with no causal or positional mask in the attention kernel.

## Files

- `qsa_sparse_gqa.py` contains the shared attention implementation and
  standalone checks. `QSASparseGQAPrefill` provides the shared base and a
  standalone dense prefill check:
  a structural fork of `fmha_v2_cutedsl/fmha.py` (same swizzled SMEM atoms,
  m16n8k16 tiled MMA, ldmatrix double-buffered BMM loops and exp2 online
  softmax) with a sparse per-row cp.async gather.
- `QSASparseGQAChunkPrefill` subclasses it for paged multi-query attention.
  Its AOT exports serve both initial and chunk prefill through the same MMA
  and online-softmax body.
- `QSASparseGQADecode` subclasses the same base for single-token split-K
  decode. It reuses the per-chunk BMM + online softmax, replaces the index
  staging and K/V row gather with paged variants (`_load_chunk_meta`,
  `_gather_stage_paged`) and runs several independent warps per CTA that
  meet only in a cooperative merge epilogue.

## Prefill kernel shapes

- Q and O have shape `[B, S, H_q, D]`, where `S` is the padded query count
  for this chunk. `query_lengths[B]` gives each sequence's valid query count.
- Initial and chunk prefill gather K/V from the pool
  `[2 * num_pages, 128, H_kv, pool_head_dim]`
  through `[B, 2, max_pages_per_seq]` page tables, with V page ids pre-offset
  by `+num_pages`. Only columns `[0, head_dim)` are read; the remaining
  columns hold QSA indexer state.
- Indices have shape `[B, S, topk]` and contain absolute KV token ids for
  each query. The attention kernel needs no past-length input: the indexer
  already includes the cached prefix when choosing each row's causal ids.
- The kernel binary-searches the packed valid-index prefix, bounded by
  `topk` because ids can refer to a cached prefix. Padded query rows skip
  the search.
- Grid `(S, H_kv, B)` — one CTA per (query token, KV head).
- M tile = the token's GQA head group (`H_q / H_kv` rows, zero-padded to
  `m_block_size = 16`); head-padding rows are never loaded or stored.
- The index list is walked in `n_block_size = 16` chunks; one warp-wave
  cp.async-gathers one 256-element K/V row (32 lanes x 128 bits). Chunk
  indices are staged in a ping-pong SMEM buffer shared by the K gather, the
  V gather and the score mask.
- `-1` entries: zero-filled K/V rows AND `-inf` scores (NaN hardening).
- Padding query rows (`t >= query_lengths[b]`) do not read Q, indices or
  K/V, and store exact zeros. All-invalid rows also store exact zeros
  (`row_sum == 0` guard).
- The softmax scale multiplies the FP32 scores at exp2 time — it is never
  pre-folded into the fp16/bf16 Q (matches the vLLM QSA numerics ordering).

Baked at compile time: `head_dim` (256), the `(Br, Bc, threads)` tuning and
`pipe_depth` (2, script default).
Runtime-dynamic: batch, query count, `H_q`, `H_kv` (group size <= 16),
`topk`, `pool_head_dim`, page geometry, softmax scale.

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
| `qsa_sparse_chunk_prefill_d256_fp16` | fp16 | initial / chunk prefill (`--chunk_prefill`) | 100, 101, 103, 110, 120, 121 |
| `qsa_sparse_chunk_prefill_d256_bf16` | bf16 | initial / chunk prefill (`--chunk_prefill`) | 100, 101, 103, 110, 120, 121 |
| `qsa_sparse_decode_d256_fp16` | fp16 | decode (`--decode --max_splits 8`) | 100, 101, 103, 110, 120, 121 |
| `qsa_sparse_decode_d256_bf16` | bf16 | decode (`--decode --max_splits 8`) | 100, 101, 103, 110, 120, 121 |

All four bake `--head_dim 256 --m_block_size 16 --n_block_size 16`; the
prefill variants use `--num_threads 32` (one warp per query token), the
decode variants `--num_threads 128 --pipe_depth 1` (4 gather warps per
CTA).

## Standalone test / benchmark (needs a GPU)

Dense initial prefill (standalone JIT):

```bash
python kernelSrcs/qsa_cutedsl/qsa_sparse_gqa.py \
    --batch_size 2 --seqlen 384 --num_head 24 --kv_group_size 12 \
    --head_dim 256 --topk 2051 --ragged
```

The in-file FP32 CuPy oracle checks the gather-softmax result against
NaN-poisoned padding and a sentinel-poisoned output buffer (padding rows must
come back as exact zeros).

Paged prefill / chunk-prefill:

```bash
python kernelSrcs/qsa_cutedsl/qsa_sparse_gqa.py --chunk_prefill \
    --batch_size 3 --seqlen 7 --past_seqlen 126 --num_head 24 \
    --kv_group_size 12 --head_dim 256 --topk 2051 --ragged
```

`--seqlen` is the new chunk's query count; `--past_seqlen` is the cached
prefix length (use `0` for initial prefill). The check generates causal
per-query index lists and compares the paged gather result against the FP32
CuPy oracle. It uses permuted pages, NaN-poisoned padding and pool tails, and
with `--ragged` includes a sequence with zero valid queries. These standalone
checks exercise attention given index lists, not the QSA indexer itself.

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

Use a fresh output directory when checking the exported variant set;
incremental builds retain existing archive members, headers and metadata.

```bash
python kernelSrcs/build_cutedsl.py --kernels qsa --gpu_arch sm_110 \
    --arch aarch64 --output_dir /tmp/qsa-aot-sm110
```

`cpp/kernels/qsaAttention/cuteDslQsaSparseRunner.{h,cpp}` consumes these
exports under `CUTE_DSL_QSA_ENABLED` (cmake
`-DENABLE_CUTE_DSL="fmha;qsa"`). Both `QsaAttentionPlugin` context phases
use the same context function, QSA indexer and `CuteDslQsaSparsePrefillRunner`.
The runner uses `qsa_sparse_chunk_prefill_d256_*` for both phase 1 (initial
prefill, zero past) and phase 2 (append at each sequence's past length).
Both read K/V directly from the paged pool. `CuteDslQsaSparseDecodeRunner` uses
`qsa_sparse_decode_d256_*` for phase 3 (single-token decode).
