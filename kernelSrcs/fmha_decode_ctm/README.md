# CTM paged GQA decode attention (`fmha_decode.py`, issue #883)

One CuTe DSL module (`cutlass.experimental.primitives`, CUTLASS 4.7, SM100 / SM101 / SM110) that replaces the XQA
decode kernels: every configuration `canCompileXQAKernel` accepts is served by an AOT export of this file.

```
build(D, kv_dtype="fp16", spec_dec=False)      # D in {32, 64, 128, 256, 512}; kv_dtype in {"fp16", "e4m3"}
python fmha_decode.py --head_dim 128 --kv_dtype e4m3 [--spec_dec] --export_only --output_dir ... \
                      --file_name ... --function_prefix ... --target_sm 110
```

Two kernel bodies share the TMA producer, online-softmax warpgroup, split-K rendezvous and epilogue helpers:

| body | AOT variants (`build_cutedsl.py`) | runtime scalars | notes |
|---|---|---|---|
| single token (`spec_dec=False`) | `ctm_decode_{f16,e4m3}_d{32,64,128,256,512}` | B, Hq, Hkv, page-table width, split plan, sliding window, attention sinks, softmax scale, k/v dequant scale, GQA ratio | fp16 ratio 9..16 = two adjacent 8-head groups per KV head (`kv_head_shift`) sharing every page fetch through L2; e4m3 uses `kind::f8f6f4` MMA with Q and P staged to e4m3, `k_scale` folded into the log2-domain QK scale, `v_scale` once on the normalised O, cluster-mailbox join for <= 8 / 4 / 0 splits at D <= 128 / 256 / 512 |
| tree / spec-decode (`spec_dec=True`) | `ctm_decode_tree_{f16,e4m3}_d*` (no e4m3 D=32, as XQA) | same + S draft tokens (`q/out [B, S, Hq, D]`), packed tree mask `[B, S, ceil(S/32)]`, both sliding-window forms (`contiguous` = XQA `CONTIGUOUS_QUERY_SWA`) | XQA `SPEC_DEC` semantics: every past key attended, draft key j iff bit j, keys at or past `seq_lens` never (mask bits past S are don't-care), no implicit causality; (token, head) rows of a group dealt into 128-row TMEM blocks (64-row blocks with an optional head-dim split on fp16 D >= 256); e4m3 shards the head dim over `D/128` sibling CTAs |

Wrapper contracts (also in the module docstring; the C++ side `static_assert`s the arity):

```
single: module, q, kv, page_table, seq_lens, sinks, o, ws_f32, ws_f16, semaphores,
        B, Hkv_groups, num_pool_pages, max_pages, gqa_per_group, splits, cluster_splits, window, has_sink,
        kv_head_shift, qk_scale, v_scale, stream,
        skip_softmax_scale_table, skip_softmax_scale_table_len                                 (module + 24)
tree:   module, q, kv, page_table, seq_lens, sinks, o, ws_f32, ws_f16, semaphores, tree_mask,
        B, Hkv, num_pool_pages, max_pages, gqa_ratio, splits, window, contiguous, has_sink, q_len, mask_words,
        row_blocks, d_split, row_block_magic, gqa_magic, split_magic, qk_scale, v_scale, stream (module + 29)
```

Skip-softmax (BLASST, single-token body): the trailing `skip_softmax_scale_table` is the same flattened int32
`(seqlen_bin, S)` table the prefill kernels take (length 0 = dense). Once per CTA the kernel floor-looks-up S by the
row's KV length and sets `lambda = S / seq_len`; the softmax warpgroup then votes per KV page — the page is dropped
when every live head has `exp(tile_max - running_max) < lambda` — and publishes the vote through an smem mbarrier
ring. The V producer (warp 3) issues a page's V TMA only after that vote (a skipped page completes its V stage with a
plain arrive, no bytes), BMM2 issues no MMA on it and its P is never written, so a skipped page costs neither the V
DRAM traffic nor the P.V. The K ring keeps DRAM busy meanwhile; V only has to land before the P ring runs dry. Every
split but the row's first prepends the first live page (the attention sink) to its K walk as a seed — scored for its
tile max only, never accumulated, no V — so its running max starts near the global one: cold per-split chains would
otherwise skip ~9% of the pages a single chain skips 65% of (18 splits, Qwen3-8B). BMM2 initialises O on the first
kept page and a split that kept nothing publishes a zero partial with m = -inf. The e4m3 16-wide ratio-16 arm runs
dense.

Invariants the runtime relies on: paged pool `[2 * NPAGES, 128, Hkv, D]` with V page ids pre-offset in row 1 of the
`[B, 2, MAXP]` page table; live splits are derived on device from `seq_lens`, so cost tracks the live KV length; the
split-K rendezvous token is a per-group generation word in device memory (a host counter would freeze into the decode
CUDA graph); `redux.sync` is emitted only for `--target_sm` 100/101 (butterfly on SM110). Bounded-SWA page tables
(retired `-1` entries) are safe: the kernels never touch a page before the window's first page.

The host dispatch lives in `cpp/kernels/decodeAttentionKernels/ctmDecodeAttentionRunner.{h,cpp}`; the attention plugin routes
every TP=1 decode to it on Blackwell (SM100/101/103/110); TP>1 ranks decode on XQA on every SM, and pre-Blackwell
SMs fall back to XQA only for the shapes the e4m3 module loses on (multi-wave launches, B*Hkv > SM count).

Provenance: each body descends from Kernel Factory campaigns run against the production XQA compiled for the same
feature set (`ctm-decode-{runtime,swa-sink,fp8kv,headdim,fp8-headdim,ratio16,tree,tree-headdim,tree-smalld,tree-fp8,
vanilla-general,fp8-general}-b200`), each validated on its campaign rows (atol/rtol 1e-2 vs the CI reference) and a
CUDA-graph replay check, then merged into this file at parity (>= 0.97x of every specialized predecessor on its rows).
