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

#pragma once

#include <cuda_runtime.h>

#if defined(CUTE_DSL_FMHA_DECODE_CTM_ENABLED)
#include <cuda.h>

#if CUDA_VERSION >= 12000 && CUDA_VERSION < 12080
typedef CUlibrary cudaLibrary_t;
extern "C" cudaError_t cudaLibraryUnload(cudaLibrary_t library);
#endif

#include "kernels/cuteDslModuleLoader.h"

#if defined(CUTE_DSL_CUDA_ERROR_CHECK)
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif
#define CUTE_DSL_CUDA_ERROR_CHECK(error) ::trt_edgellm::detail::recordCuteDslCudaError(static_cast<cudaError_t>(error))
#include "cutedsl_all.h"
#undef CUTE_DSL_CUDA_ERROR_CHECK
#endif // defined(CUTE_DSL_FMHA_DECODE_CTM_ENABLED)

#include "common/specDecodeLimits.h"

#include <cstddef>
#include <cstdint>

namespace trt_edgellm
{

//! Launch parameters for the CTM (CuTe DSL experimental.primitives) paged GQA decode kernels.
//! Fixed contract: fp16 Q/O, kTOKENS_PER_PAGE tokens per page, page table row 1 pre-offset. qSeqLen == 1 is
//! vanilla decode; qSeqLen > 1 is tree decode over the last qSeqLen cache positions (XQA SPEC_DEC semantics:
//! every past key attended, draft keys per treeMask bit, no implicit causality).
struct CtmDecodeParams
{
    void* output{nullptr};             //!< [B, qSeqLen, Hq, D] fp16
    void const* q{nullptr};            //!< [B, qSeqLen, Hq, D] fp16, RoPE applied
    void* kvPool{nullptr};             //!< Paged pool [2 * numPages, tokensPerPage, Hkv, D], fp16 or e4m3
    int32_t const* pageTable{nullptr}; //!< [B, 2, maxPagesPerSeq]; row 1 holds V page ids pre-offset by numPages
    int32_t const* seqLens{nullptr};   //!< [B] valid KV length per sequence (tree: past + qSeqLen)
    float const* sinks{nullptr};       //!< [Hq] fp32 attention-sink logits, or nullptr
    int32_t const* treeMask{nullptr};  //!< [B, qSeqLen, maskWords] int32; bit (j % 32) of word (j / 32) of row i
                                       //!< set = draft query i attends draft key j. Bits at j >= qSeqLen may hold
                                       //!< anything: keys at or past seqLens[b] are never attended. Tree decode only.
    void* workspace{nullptr};          //!< Split-K scratch of workspaceElems() / treeWorkspaceElems() fp32 elements
    int32_t* semaphores{nullptr};      //!< semaphoreElems() int32 split-K counters; zero at first use, self-resetting
    int32_t batchSize{0};
    int32_t qSeqLen{1};             //!< Query tokens per sequence
    int32_t maskWords{0};           //!< treeMask row length in int32 words
    bool contiguousQuerySwa{false}; //!< Tree + window: per-row left edge (XQA CONTIGUOUS_QUERY_SWA) vs shared edge
    int32_t numQHeads{0};
    int32_t numKVHeads{0};
    int32_t headSize{0};
    int32_t numPages{0};          //!< K page count of the pool (the pool holds 2 * numPages)
    int32_t maxPagesPerSeq{0};    //!< Page-table row length
    int32_t slidingWindow{0};     //!< Keys attended (current token included); 0 = full span
    float attentionScale{1.F};    //!< QK^T multiplier (the plugin's attention_scale, not assumed 1/sqrt(D))
    bool fp8KVCache{false};       //!< Pool element type is e4m3; kScale / vScale apply
    float kScale{1.F};            //!< K dequant scale (quantised -> original)
    float vScale{1.F};            //!< V dequant scale (quantised -> original)
    bool fallbackAvailable{true}; //!< The caller has another decode backend; run() may decline slow shapes
    //! Skip-softmax (BLASST) scale table: flattened int32 (seqlen_bin, S) pairs, bins ascending. The kernel
    //! floor-looks-up S by each sequence's KV length and drops a KV page when every head of the group has
    //! exp(tile_max - running_max) < S / seq_len. nullptr / 0 = dense. Single-token decode only.
    int32_t const* skipSoftmaxScaleTable{nullptr};
    int32_t skipSoftmaxScaleTableLen{0};
    cudaStream_t stream{nullptr};
};

//! Dispatch shim for the runtime-shaped CTM decode kernels (kernelSrcs/fmha_decode_ctm/fmha_decode.py): one AOT
//! module per (spec_dec, KV dtype, head_dim) serves every (B, Hq, Hkv, page capacity, window, sinks, GQA ratio);
//! the split-K plan is chosen per launch.
class CtmDecodeAttentionRunner
{
public:
    static constexpr int32_t kMaxGqaRatio = 8;     //!< fp16 single-token decode below D == 128 (XQA's envelope)
    static constexpr int32_t kMaxGqaRatioR16 = 16; //!< fp16 D >= 128, every e4m3 module, tree decode
    //! Tree query tokens per sequence: the shared speculative-decoding tree budget. The kernel itself reads
    //! ceil(qSeqLen / 32) mask words and has no bound of its own.
    static constexpr int32_t kMaxTreeQuerySeqLen = kMAX_SPEC_VERIFY_TREE_TOKENS;

    //! \returns true when a module exists for the configuration: fp16 or e4m3 KV at D in {32, 64, 128, 256, 512}
    //! with sliding window and sinks; GQA ratio 1..8 on fp16 below D == 128, 1..16 otherwise.
    static bool supported(
        int32_t numQHeads, int32_t numKVHeads, int32_t headSize, bool fp8KVCache, bool slidingWindow, bool sinks);

    //! fp32 element count of the split-K scratch at the widest plan for this configuration.
    static size_t workspaceElems(int32_t batchSize, int32_t numQHeads, int32_t headSize);

    //! int32 element count of the split-K semaphore array.
    static size_t semaphoreElems(int32_t batchSize, int32_t numKVHeads);

    //! \returns true when a tree (spec-decode) module serves the configuration: fp16 KV at D in {32, 64, 128, 256,
    //! 512} or e4m3 KV at D in {64, 128, 256, 512}, GQA ratio 1..16, sinks, and either sliding-window form (per-row
    //! XQA CONTIGUOUS_QUERY_SWA or the shared left edge) as runtime flags.
    static bool supportedTree(int32_t numQHeads, int32_t numKVHeads, int32_t headSize, bool fp8KVCache,
        bool slidingWindow, bool contiguousQuerySwa);

    //! \returns true when the tree modules apply every treeMask bit of a qSeqLen-token tree (1..kMaxTreeQuerySeqLen).
    static bool supportsTreeQuerySeqLen(int32_t qSeqLen);

    //! fp32 element count of the tree module's split-K scratch at the widest plan.
    static size_t treeWorkspaceElems(
        int32_t batchSize, int32_t qSeqLen, int32_t numQHeads, int32_t numKVHeads, int32_t headSize);

    //! int32 element count of the tree module's semaphores: one stride per (batch, kv-head, row block, head shard).
    static size_t treeSemaphoreElems(int32_t batchSize, int32_t qSeqLen, int32_t numQHeads, int32_t numKVHeads);

    //! Launch the kernel. Precondition: supported(...) returned true.
    //! \returns false when the module could not be loaded or the launch failed; the caller falls back.
    static bool run(CtmDecodeParams const& params);
};

} // namespace trt_edgellm
