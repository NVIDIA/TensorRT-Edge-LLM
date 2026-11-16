/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ctmDecodeAttentionRunner.h"

#include "common/cudaUtils.h"
#include "common/logger.h"

#include <algorithm>
#include <cmath>

#if defined(CUTE_DSL_FMHA_DECODE_CTM_ENABLED)
#include "kernels/contextAttentionKernels/cuteDslTensorDescriptors.h"
#endif

namespace trt_edgellm
{

namespace
{

//! Single-token modules pad the head axis by one 16-head tile so the join reads a full tile at any GQA ratio.
constexpr int32_t kHeadPad = 16;
//! Widest single-token split-K plan (fp16 and e4m3 alike).
constexpr int32_t kMaxSplits = 64;
//! e4m3 single-token plan: 32 splits unless the engine is long (> 64 pages) at D == 512.
constexpr int32_t kE4m3NarrowSplits = 32;
constexpr int32_t kE4m3WideEnginePages = 64;
//! Per-group semaphore stride of the single-token modules: e4m3 keeps one rendezvous flag word per split plus a
//! generation word in a 128-word stride; fp16 uses 2 counters inside the same stride.
constexpr int32_t kSemaphoresPerGroup = 128;
//! Tree modules: the (draft token, q-head) rows of a (batch, kv-head) group are dealt into 128-row blocks (64-row
//! blocks on fp16 D >= 256); each block owns a 32-word semaphore stride; the join folds at most 16 splits
//! (18 on fp16 D == 512).
constexpr int32_t kTreeRowsTile = 128;
constexpr int32_t kTreeWideRowsTile = 64;
constexpr int32_t kTreeSemStride = 32;
constexpr int32_t kTreeMaxSplits = 16;
constexpr int32_t kTreeMaxSplitsF16D512 = 18;
constexpr int32_t kTreeNarrowSplits = 8;

//! e4m3 tree modules shard the head dim over D / 128 sibling CTAs so the O accumulator fits TMEM next to the S ring.
int32_t treeShardWays(int32_t headSize, bool fp8KVCache)
{
    return fp8KVCache ? std::max(1, headSize / 128) : 1;
}

int32_t treeRowBlocks(int32_t qSeqLen, int32_t gqaRatio, int32_t rowsTile)
{
    return (qSeqLen * gqaRatio + rowsTile - 1) / rowsTile;
}

int32_t roundUp4(int32_t x)
{
    return (x + 3) / 4 * 4;
}

#if defined(CUTE_DSL_FMHA_DECODE_CTM_ENABLED)
//! fp16 tree modules at D >= 256 use the 64-row tiling with an optional head-dim split at D == 512.
bool treeWide(int32_t headSize, bool fp8KVCache)
{
    return headSize >= 256 && !fp8KVCache;
}

//! ceil(2^16 / d): the tree kernel divides block indices by row blocks / GQA ratio / splits with one IMAD and a
//! shift instead of an integer division; exact for the launch ranges (d <= 1024, quotients < 2^16 / d).
int32_t ceilDivMagic(int32_t d)
{
    return (65536 + d - 1) / d;
}

//! One CTA per SM dealt across the groups, never wider than the pages a sequence can hold or the module's join
//! width. Live splits are derived on device from seq_lens.
int32_t planSplits(int32_t groups, int32_t maxPagesPerSeq, int32_t maxSplits)
{
    int32_t const splits = std::min({getDeviceMultiProcessorCount() / groups, maxPagesPerSeq, maxSplits});
    return std::max(splits, 1);
}

//! A window bounds the live span to (W / page) + 2 pages whatever the sequence length.
int32_t windowPages(int32_t slidingWindow)
{
    return (slidingWindow >> 7) + 2;
}

// One AOT module per (spec_dec, KV dtype, head_dim); every module of a family shares one wrapper contract.
// clang-format off
detail::LazyKernelModule<ctm_decode_f16_d32_Kernel_Module_t> sSingleF16D32{};
detail::LazyKernelModule<ctm_decode_f16_d64_Kernel_Module_t> sSingleF16D64{};
detail::LazyKernelModule<ctm_decode_f16_d128_Kernel_Module_t> sSingleF16D128{};
detail::LazyKernelModule<ctm_decode_f16_d256_Kernel_Module_t> sSingleF16D256{};
detail::LazyKernelModule<ctm_decode_f16_d512_Kernel_Module_t> sSingleF16D512{};
detail::LazyKernelModule<ctm_decode_e4m3_d32_Kernel_Module_t> sSingleE4m3D32{};
detail::LazyKernelModule<ctm_decode_e4m3_d64_Kernel_Module_t> sSingleE4m3D64{};
detail::LazyKernelModule<ctm_decode_e4m3_d128_Kernel_Module_t> sSingleE4m3D128{};
detail::LazyKernelModule<ctm_decode_e4m3_d256_Kernel_Module_t> sSingleE4m3D256{};
detail::LazyKernelModule<ctm_decode_e4m3_d512_Kernel_Module_t> sSingleE4m3D512{};
detail::LazyKernelModule<ctm_decode_tree_f16_d32_Kernel_Module_t> sTreeF16D32{};
detail::LazyKernelModule<ctm_decode_tree_f16_d64_Kernel_Module_t> sTreeF16D64{};
detail::LazyKernelModule<ctm_decode_tree_f16_d128_Kernel_Module_t> sTreeF16D128{};
detail::LazyKernelModule<ctm_decode_tree_f16_d256_Kernel_Module_t> sTreeF16D256{};
detail::LazyKernelModule<ctm_decode_tree_f16_d512_Kernel_Module_t> sTreeF16D512{};
detail::LazyKernelModule<ctm_decode_tree_e4m3_d64_Kernel_Module_t> sTreeE4m3D64{};
detail::LazyKernelModule<ctm_decode_tree_e4m3_d128_Kernel_Module_t> sTreeE4m3D128{};
detail::LazyKernelModule<ctm_decode_tree_e4m3_d256_Kernel_Module_t> sTreeE4m3D256{};
detail::LazyKernelModule<ctm_decode_tree_e4m3_d512_Kernel_Module_t> sTreeE4m3D512{};
// clang-format on

static_assert(cutedsl::WrapperArity<decltype(&cute_dsl_ctm_decode_f16_d32_wrapper)>::value == 25
        && cutedsl::WrapperArity<decltype(&cute_dsl_ctm_decode_f16_d64_wrapper)>::value == 25
        && cutedsl::WrapperArity<decltype(&cute_dsl_ctm_decode_f16_d128_wrapper)>::value == 25
        && cutedsl::WrapperArity<decltype(&cute_dsl_ctm_decode_f16_d256_wrapper)>::value == 25
        && cutedsl::WrapperArity<decltype(&cute_dsl_ctm_decode_f16_d512_wrapper)>::value == 25
        && cutedsl::WrapperArity<decltype(&cute_dsl_ctm_decode_e4m3_d32_wrapper)>::value == 25
        && cutedsl::WrapperArity<decltype(&cute_dsl_ctm_decode_e4m3_d64_wrapper)>::value == 25
        && cutedsl::WrapperArity<decltype(&cute_dsl_ctm_decode_e4m3_d128_wrapper)>::value == 25
        && cutedsl::WrapperArity<decltype(&cute_dsl_ctm_decode_e4m3_d256_wrapper)>::value == 25
        && cutedsl::WrapperArity<decltype(&cute_dsl_ctm_decode_e4m3_d512_wrapper)>::value == 25,
    "ctm_decode_<dtype>_d<N> wrapper: expected (module, q, kv, page_table, seq_lens, sinks, o, ws_f32, ws_f16, "
    "semaphores, B, Hkv_groups, num_pool_pages, max_pages, gqa_per_group, splits, cluster_splits, window, has_sink, "
    "kv_head_shift, qk_scale, v_scale, stream, skip_softmax_scale_table, skip_softmax_scale_table_len).");

static_assert(cutedsl::WrapperArity<decltype(&cute_dsl_ctm_decode_tree_f16_d32_wrapper)>::value == 30
        && cutedsl::WrapperArity<decltype(&cute_dsl_ctm_decode_tree_f16_d64_wrapper)>::value == 30
        && cutedsl::WrapperArity<decltype(&cute_dsl_ctm_decode_tree_f16_d128_wrapper)>::value == 30
        && cutedsl::WrapperArity<decltype(&cute_dsl_ctm_decode_tree_f16_d256_wrapper)>::value == 30
        && cutedsl::WrapperArity<decltype(&cute_dsl_ctm_decode_tree_f16_d512_wrapper)>::value == 30
        && cutedsl::WrapperArity<decltype(&cute_dsl_ctm_decode_tree_e4m3_d64_wrapper)>::value == 30
        && cutedsl::WrapperArity<decltype(&cute_dsl_ctm_decode_tree_e4m3_d128_wrapper)>::value == 30
        && cutedsl::WrapperArity<decltype(&cute_dsl_ctm_decode_tree_e4m3_d256_wrapper)>::value == 30
        && cutedsl::WrapperArity<decltype(&cute_dsl_ctm_decode_tree_e4m3_d512_wrapper)>::value == 30,
    "ctm_decode_tree_<dtype>_d<N> wrapper: expected (module, q, kv, page_table, seq_lens, sinks, o, ws_f32, ws_f16, "
    "semaphores, tree_mask, B, Hkv, num_pool_pages, max_pages, gqa_ratio, splits, window, contiguous, has_sink, q_len, "
    "mask_words, row_blocks, d_split, row_block_magic, gqa_magic, split_magic, qk_scale, v_scale, stream).");

//! Single-token launch. fp16 serves GQA ratio 9..16 as two adjacent 8-head groups per KV head (kv_head_shift = 1)
//! that share every page fetch through L2; e4m3 keeps one 16-head group and joins ratio-16 groups through global
//! memory (the cluster mailbox holds 8 / 4 / 0 split slots at D <= 128 / 256 / 512 and an 8-head slot).
template <auto Wrapper, auto Loader, auto Unloader, typename Module>
bool runSingle(detail::LazyKernelModule<Module>& state, char const* name, CtmDecodeParams const& p)
{
    if (!detail::ensureModuleLoaded<Loader, Unloader>(state, name, p.stream))
    {
        return false;
    }
    int32_t const gqaRatio = p.numQHeads / p.numKVHeads;
    int32_t const headShift = (!p.fp8KVCache && gqaRatio > CtmDecodeAttentionRunner::kMaxGqaRatio) ? 1 : 0;
    int32_t const groups = p.numKVHeads << headShift;
    int32_t splits = 1;
    int32_t clusterSplits = 1;
    float qkScale = p.attentionScale;
    if (p.fp8KVCache)
    {
        int32_t const cap
            = (p.headSize == 512 && p.maxPagesPerSeq > kE4m3WideEnginePages) ? kMaxSplits : kE4m3NarrowSplits;
        splits = planSplits(p.batchSize * groups, p.maxPagesPerSeq, cap);
        if (p.slidingWindow > 0)
        {
            splits = std::max(std::min(splits, windowPages(p.slidingWindow)), 1);
        }
        int32_t const clusterCap = p.headSize <= 128 ? 8 : (p.headSize == 256 ? 4 : 1);
        clusterSplits = (splits <= clusterCap && gqaRatio <= CtmDecodeAttentionRunner::kMaxGqaRatio) ? splits : 1;
        // K scale folds into the log2-domain QK scale exactly as XQA folds kCacheScale; V scale is applied once on
        // the normalised O.
        qkScale = p.kScale * p.attentionScale * static_cast<float>(M_LOG2E);
    }
    else
    {
        splits = planSplits(p.batchSize * groups, p.maxPagesPerSeq, kMaxSplits);
        if (p.slidingWindow > 0)
        {
            splits = std::max(std::min(splits, windowPages(p.slidingWindow)), 1);
        }
        else if (p.headSize == 512 && p.batchSize * groups != 1)
        {
            // The D = 512 plan halves its live splits on device once the span fits one window per split.
            splits = std::min(splits, std::max((p.maxPagesPerSeq + 1) >> 1, 8));
        }
        // Join-width tiering: a narrower launch that keeps the same deepest window folds fewer partials.
        if (splits > 16)
        {
            int32_t const depth = (p.maxPagesPerSeq + splits - 1) / splits;
            if ((p.maxPagesPerSeq + 15) / 16 == depth)
            {
                splits = 16;
            }
            else if (splits > 32 && (p.maxPagesPerSeq + 31) / 32 == depth)
            {
                splits = 32;
            }
        }
    }
    void* sinks = p.sinks != nullptr ? const_cast<float*>(p.sinks) : const_cast<void*>(p.q);
    // The wrapper always takes a valid table pointer; the kernel sizes its view by the length (0 = dense).
    bool const skip = p.skipSoftmaxScaleTable != nullptr && p.skipSoftmaxScaleTableLen >= 2;
    void* const skipTable = skip ? const_cast<int32_t*>(p.skipSoftmaxScaleTable) : const_cast<int32_t*>(p.seqLens);
    int32_t const rc = Wrapper(&state.module, const_cast<void*>(p.q), p.kvPool, const_cast<int32_t*>(p.pageTable),
        const_cast<int32_t*>(p.seqLens), sinks, p.output, p.workspace, p.workspace, p.semaphores, p.batchSize, groups,
        2 * p.numPages, p.maxPagesPerSeq, gqaRatio >> headShift, splits, clusterSplits, p.slidingWindow,
        p.sinks != nullptr ? 1 : 0, headShift, qkScale, p.vScale, p.stream, skipTable,
        skip ? p.skipSoftmaxScaleTableLen : 0);
    if (rc != 0)
    {
        LOG_ERROR("CtmDecodeAttentionRunner: %s launch failed (rc=%d).", name, rc);
    }
    return rc == 0;
}

//! Tree launch: row blocks already fill the machine, so the split-K deal is cut against them and the head-dim
//! shards; below 4 * narrow pages of reach a plan past the narrow deal cannot save the two pages of depth its
//! wider join costs.
template <auto Wrapper, auto Loader, auto Unloader, typename Module>
bool runTree(detail::LazyKernelModule<Module>& state, char const* name, CtmDecodeParams const& p)
{
    if (!detail::ensureModuleLoaded<Loader, Unloader>(state, name, p.stream))
    {
        return false;
    }
    int32_t const gqaRatio = p.numQHeads / p.numKVHeads;
    bool const wide = treeWide(p.headSize, p.fp8KVCache);
    int32_t const ways = treeShardWays(p.headSize, p.fp8KVCache);
    int32_t const rowBlocks = treeRowBlocks(p.qSeqLen, gqaRatio, wide ? kTreeWideRowsTile : kTreeRowsTile);
    int32_t const maxSplits = (wide && p.headSize == 512) ? kTreeMaxSplitsF16D512 : kTreeMaxSplits;
    int32_t splits = planSplits(p.batchSize * p.numKVHeads * rowBlocks * ways, p.maxPagesPerSeq, maxSplits);
    int32_t reach = p.maxPagesPerSeq;
    if (wide && p.slidingWindow > 0)
    {
        reach = std::min(reach, windowPages(p.slidingWindow));
    }
    if (reach <= 4 * kTreeNarrowSplits && splits > kTreeNarrowSplits)
    {
        splits = kTreeNarrowSplits;
    }
    // d_split: fp16 D = 512 takes a second CTA per (group, split) for half the head dim when the machine has room;
    // e4m3 always shards into `ways` CTAs and passes log2(ways).
    int32_t dSplit = 0;
    if (p.fp8KVCache)
    {
        dSplit = ways == 4 ? 2 : (ways == 2 ? 1 : 0);
    }
    else if (wide && p.headSize == 512
        && 2 * p.batchSize * p.numKVHeads * rowBlocks * splits <= getDeviceMultiProcessorCount())
    {
        dSplit = 1;
    }
    // The tree kernels fold log2e on device; e4m3 folds the K scale into the softmax scale exactly as XQA does.
    float const qkScale = p.fp8KVCache ? p.kScale * p.attentionScale : p.attentionScale;
    void* sinks = p.sinks != nullptr ? const_cast<float*>(p.sinks) : const_cast<void*>(p.q);
    int32_t const rc = Wrapper(&state.module, const_cast<void*>(p.q), p.kvPool, const_cast<int32_t*>(p.pageTable),
        const_cast<int32_t*>(p.seqLens), sinks, p.output, p.workspace, p.workspace, p.semaphores,
        const_cast<int32_t*>(p.treeMask), p.batchSize, p.numKVHeads, 2 * p.numPages, p.maxPagesPerSeq, gqaRatio, splits,
        p.slidingWindow, p.contiguousQuerySwa ? 1 : 0, p.sinks != nullptr ? 1 : 0, p.qSeqLen, p.maskWords, rowBlocks,
        dSplit, ceilDivMagic(rowBlocks), ceilDivMagic(gqaRatio), ceilDivMagic(splits), qkScale, p.vScale, p.stream);
    if (rc != 0)
    {
        LOG_ERROR("CtmDecodeAttentionRunner: %s launch failed (rc=%d).", name, rc);
    }
    return rc == 0;
}

#define CTM_RUN(fn, tag, state)                                                                                        \
    fn<cute_dsl_##tag##_wrapper, tag##_Kernel_Module_Load, tag##_Kernel_Module_Unload>(state, #tag, p)

bool runSingleModule(CtmDecodeParams const& p)
{
    if (p.fp8KVCache)
    {
        switch (p.headSize)
        {
        case 32: return CTM_RUN(runSingle, ctm_decode_e4m3_d32, sSingleE4m3D32);
        case 64: return CTM_RUN(runSingle, ctm_decode_e4m3_d64, sSingleE4m3D64);
        case 128: return CTM_RUN(runSingle, ctm_decode_e4m3_d128, sSingleE4m3D128);
        case 256: return CTM_RUN(runSingle, ctm_decode_e4m3_d256, sSingleE4m3D256);
        case 512: return CTM_RUN(runSingle, ctm_decode_e4m3_d512, sSingleE4m3D512);
        default: return false;
        }
    }
    switch (p.headSize)
    {
    case 32: return CTM_RUN(runSingle, ctm_decode_f16_d32, sSingleF16D32);
    case 64: return CTM_RUN(runSingle, ctm_decode_f16_d64, sSingleF16D64);
    case 128: return CTM_RUN(runSingle, ctm_decode_f16_d128, sSingleF16D128);
    case 256: return CTM_RUN(runSingle, ctm_decode_f16_d256, sSingleF16D256);
    case 512: return CTM_RUN(runSingle, ctm_decode_f16_d512, sSingleF16D512);
    default: return false;
    }
}

bool runTreeModule(CtmDecodeParams const& p)
{
    if (p.fp8KVCache)
    {
        switch (p.headSize)
        {
        case 64: return CTM_RUN(runTree, ctm_decode_tree_e4m3_d64, sTreeE4m3D64);
        case 128: return CTM_RUN(runTree, ctm_decode_tree_e4m3_d128, sTreeE4m3D128);
        case 256: return CTM_RUN(runTree, ctm_decode_tree_e4m3_d256, sTreeE4m3D256);
        case 512: return CTM_RUN(runTree, ctm_decode_tree_e4m3_d512, sTreeE4m3D512);
        default: return false;
        }
    }
    switch (p.headSize)
    {
    case 32: return CTM_RUN(runTree, ctm_decode_tree_f16_d32, sTreeF16D32);
    case 64: return CTM_RUN(runTree, ctm_decode_tree_f16_d64, sTreeF16D64);
    case 128: return CTM_RUN(runTree, ctm_decode_tree_f16_d128, sTreeF16D128);
    case 256: return CTM_RUN(runTree, ctm_decode_tree_f16_d256, sTreeF16D256);
    case 512: return CTM_RUN(runTree, ctm_decode_tree_f16_d512, sTreeF16D512);
    default: return false;
    }
}

#undef CTM_RUN
#endif

[[maybe_unused]] bool headSizeServed(int32_t headSize)
{
    return headSize == 32 || headSize == 64 || headSize == 128 || headSize == 256 || headSize == 512;
}

} // namespace

bool CtmDecodeAttentionRunner::supported(
    int32_t numQHeads, int32_t numKVHeads, int32_t headSize, bool fp8KVCache, bool slidingWindow, bool sinks)
{
#if defined(CUTE_DSL_FMHA_DECODE_CTM_ENABLED)
    if (numKVHeads <= 0 || numQHeads % numKVHeads != 0)
    {
        return false;
    }
    int32_t const ratio = numQHeads / numKVHeads;
    // XQA's envelope: GQA ratio 9..16 only at D >= 128 on fp16, at every D on e4m3.
    bool const ratio16 = fp8KVCache || headSize >= 128;
    if (ratio < 1 || ratio > (ratio16 ? kMaxGqaRatioR16 : kMaxGqaRatio))
    {
        return false;
    }
    (void) slidingWindow; // runtime scalars on every module
    (void) sinks;
    return headSizeServed(headSize);
#else
    (void) numQHeads;
    (void) numKVHeads;
    (void) headSize;
    (void) fp8KVCache;
    (void) slidingWindow;
    (void) sinks;
    return false;
#endif
}

size_t CtmDecodeAttentionRunner::workspaceElems(int32_t batchSize, int32_t numQHeads, int32_t headSize)
{
    // [B*Hq + pad] x [splits*D partial O | splits running max | splits colsum] at the widest plan.
    return static_cast<size_t>(batchSize * numQHeads + kHeadPad) * (kMaxSplits * headSize + 2 * kMaxSplits);
}

size_t CtmDecodeAttentionRunner::semaphoreElems(int32_t batchSize, int32_t numKVHeads)
{
    // fp16 ratio 9..16 runs two groups per KV head.
    return static_cast<size_t>(kSemaphoresPerGroup) * batchSize * numKVHeads * 2;
}

bool CtmDecodeAttentionRunner::supportedTree(int32_t numQHeads, int32_t numKVHeads, int32_t headSize, bool fp8KVCache,
    bool slidingWindow, bool contiguousQuerySwa)
{
#if defined(CUTE_DSL_FMHA_DECODE_CTM_ENABLED)
    if (numKVHeads <= 0 || numQHeads % numKVHeads != 0)
    {
        return false;
    }
    int32_t const ratio = numQHeads / numKVHeads;
    if (ratio < 1 || ratio > kMaxGqaRatioR16)
    {
        return false;
    }
    (void) slidingWindow; // both window forms are runtime flags on every tree module
    (void) contiguousQuerySwa;
    if (fp8KVCache && headSize == 32)
    {
        return false; // XQA has no e4m3 SPEC_DEC kernel at D = 32 either
    }
    return headSizeServed(headSize);
#else
    (void) numQHeads;
    (void) numKVHeads;
    (void) headSize;
    (void) fp8KVCache;
    (void) slidingWindow;
    (void) contiguousQuerySwa;
    return false;
#endif
}

size_t CtmDecodeAttentionRunner::treeWorkspaceElems(
    int32_t batchSize, int32_t qSeqLen, int32_t numQHeads, int32_t numKVHeads, int32_t headSize)
{
    // Rows x [splits * D/ways partial O | splits running max | splits colsum (padded to 4)] at the widest plan of
    // either KV dtype: the dense 128-row block space [B * Hkv * RB * ways * 128] (fp16 D <= 128, every e4m3
    // module) or the fp16 wide layout [B*S*Hq + one 64-row tile] with up to 18 splits.
    int32_t const ratio = numQHeads / numKVHeads;
    int32_t const ways = treeShardWays(headSize, true);
    size_t const denseRows
        = static_cast<size_t>(batchSize) * numKVHeads * treeRowBlocks(qSeqLen, ratio, kTreeRowsTile) * kTreeRowsTile;
    size_t const dense = std::max(denseRows * (kTreeMaxSplits * headSize + 2 * roundUp4(kTreeMaxSplits)),
        denseRows * ways * (kTreeMaxSplits * (headSize / ways) + 2 * roundUp4(kTreeMaxSplits)));
    if (headSize <= 128)
    {
        return dense;
    }
    int32_t const wideSplits = headSize == 512 ? kTreeMaxSplitsF16D512 : kTreeMaxSplits;
    size_t const wide = static_cast<size_t>(batchSize * qSeqLen * numQHeads + kTreeWideRowsTile)
        * (wideSplits * headSize + 2 * roundUp4(wideSplits));
    return std::max(dense, wide);
}

size_t CtmDecodeAttentionRunner::treeSemaphoreElems(
    int32_t batchSize, int32_t qSeqLen, int32_t numQHeads, int32_t numKVHeads)
{
    // One stride per (batch, kv-head, row block, head-dim shard): fp16 D >= 256 deals 64-row blocks and may split
    // the head dim in two; e4m3 D = 512 deals 128-row blocks over four shards.
    int32_t const ratio = numQHeads / numKVHeads;
    int32_t const groups = std::max(
        2 * treeRowBlocks(qSeqLen, ratio, kTreeWideRowsTile), 4 * treeRowBlocks(qSeqLen, ratio, kTreeRowsTile));
    return static_cast<size_t>(kTreeSemStride) * batchSize * numKVHeads * groups;
}

bool CtmDecodeAttentionRunner::supportsTreeQuerySeqLen(int32_t qSeqLen)
{
    return qSeqLen >= 1 && qSeqLen <= kMaxTreeQuerySeqLen;
}

bool CtmDecodeAttentionRunner::run(CtmDecodeParams const& p)
{
#if defined(CUTE_DSL_FMHA_DECODE_CTM_ENABLED)
    if (p.qSeqLen > 1 || p.treeMask != nullptr)
    {
        if (!supportsTreeQuerySeqLen(p.qSeqLen))
        {
            if (!p.fallbackAvailable)
            {
                LOG_ERROR("CtmDecodeAttentionRunner: tree decode with %d query tokens exceeds the supported %d.",
                    p.qSeqLen, kMaxTreeQuerySeqLen);
            }
            return false;
        }
        return runTreeModule(p);
    }
    // The e4m3 single-token module loses to XQA once the (batch, kv-head) groups exceed the SM count (multi-wave
    // launches): measured 0.41x-0.99x on Thor's 20 SMs at B*Hkv >= 32 while single-wave rows win 1.3x-1.75x.
    // Callers with another backend take it there.
    if (p.fp8KVCache && p.fallbackAvailable && p.batchSize * p.numKVHeads > getDeviceMultiProcessorCount())
    {
        return false;
    }
    return runSingleModule(p);
#else
    (void) p;
    return false;
#endif
}

} // namespace trt_edgellm
