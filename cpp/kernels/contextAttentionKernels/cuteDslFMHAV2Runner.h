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

#include "cuteDslFmhaTypes.h"

#include <cuda_runtime.h>

#if defined(CUTE_DSL_FMHA_ENABLED)
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
#endif // defined(CUTE_DSL_FMHA_ENABLED)

#include "common/dataType.h"
#include <climits>
#include <cstdint>

namespace trt_edgellm
{

//! Runner for the CuTe DSL FMHA-v2 kernels.
//!
//! Dense LLM kernels consume separate BSND Q/K/V tensors, while native-paged LLM kernels consume
//! BSND or packed Q/O and the Edge-LLM NHD paged KV pool. Vision blocks support both dense separate
//! Q/K/V and packed Q/O with paged KV; padding uses dense Q/K/V and ViT uses packed separate Q/K/V. The
//! runner remains separate from CuteDslFMHARunner because the optimized SM100/101/103/110 family has a
//! distinct ABI.
class CuteDslFMHAV2Runner
{
public:
    CuteDslFMHAV2Runner(int32_t numQHeads, int32_t numKVHeads, int32_t headDim, int32_t batchSize = 0,
        int32_t seqLenQ = 0, int32_t kvSeqLen = 0, bool useSmallD64 = true);

    ~CuteDslFMHAV2Runner() = default;
    CuteDslFMHAV2Runner(CuteDslFMHAV2Runner const&) = delete;
    CuteDslFMHAV2Runner& operator=(CuteDslFMHAV2Runner const&) = delete;

    //! Returns whether the target AOT family covers this dense context shape.
    static bool canImplement(int32_t numQHeads, int32_t numKVHeads, int32_t headSize, int32_t smVersion,
        trt_edgellm::DataType dataType, CuteDslFMHAV2MaskType maskType);

    //! Returns whether the target AOT family covers native FP16 Edge-LLM paged KV for this context shape.
    static bool canImplementPaged(int32_t numQHeads, int32_t numKVHeads, int32_t headSize, int32_t smVersion,
        trt_edgellm::DataType dataType, CuteDslFMHAV2MaskType maskType);

    //! Returns whether the target AOT family covers packed-Q/O native paged attention.
    static bool canImplementPagedRagged(int32_t numQHeads, int32_t numKVHeads, int32_t headSize, int32_t smVersion,
        trt_edgellm::DataType dataType, CuteDslFMHAV2MaskType maskType);

    //! Returns whether the target AOT family covers this packed ViT shape.
    static bool canImplementViT(int32_t headSize, int32_t smVersion, trt_edgellm::DataType dataType);

    //! Ensures the exact causal or sliding-causal LLM variant selected by run() is loaded.
    bool preflightLlm(cudaStream_t stream, int32_t slidingWindowSize = INT_MAX);

    //! Ensures the exact native-paged causal or sliding-causal LLM variant selected by runPaged() is loaded (the
    //! scale table does not select a variant).
    bool preflightPaged(cudaStream_t stream, int32_t slidingWindowSize = INT_MAX,
        int32_t const* skipSoftmaxScaleTable = nullptr, int32_t skipSoftmaxScaleTableLen = 0);

    //! Loads the causal, sliding-causal, or vision-block variant selected by runPagedRagged().
    bool preflightPagedRagged(cudaStream_t stream, int32_t slidingWindowSize = INT_MAX, bool useBidirectional = false);

    //! Ensures the dense non-causal padding variant selected by runPadding() is loaded.
    bool preflightPadding(cudaStream_t stream);

    //! Ensures the vision-block variant selected by runVisionBlock() is loaded.
    bool preflightVisionBlock(cudaStream_t stream);

    //! Ensures the exact packed ViT variant selected by run() is loaded.
    bool preflightViT(cudaStream_t stream);

    //! Runs causal or sliding-causal LLM context attention over dense FP16 K/V.
    bool run(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr, int32_t const* cuKVSeqLens,
        cudaStream_t stream, float attentionScale, int32_t slidingWindowSize = INT_MAX);

    //! Runs FP16 causal or sliding-causal attention directly against an FP16 NHD paged KV pool.
    //! @param skipSoftmaxScaleTable Skip-softmax (BLASST) scale table, device Int32 of flattened (seqlen_bin, S)
    //!        pairs with bins ascending; a flat S is the one-row table {0, S}. Causal, non-sliding only: per
    //!        sequence the kernel floor-looks-up S by its KV length (cuKVSeqLens, which is never padded here) and
    //!        skips the P*V GEMM of KV tiles whose contribution is below lambda = S / seqlen_kv of the running
    //!        max — approximate. Sequences below the first bin or with S >= seqlen_kv run dense.
    //! @param skipSoftmaxScaleTableLen Element count of the table; 0 (the default) runs every sequence dense.
    bool runPaged(void const* qPtr, void const* pagedKVPoolPtr, int32_t const* kvCachePageList, void* oPtr,
        int32_t const* cuQSeqLens, int32_t const* cuKVSeqLens, int32_t numFlatPages, int32_t maxPagesPerSeq,
        int32_t tokensPerPage, cudaStream_t stream, float attentionScale, int32_t slidingWindowSize = INT_MAX,
        int32_t const* skipSoftmaxScaleTable = nullptr, int32_t skipSoftmaxScaleTableLen = 0);

    //! Load the separate-prefix variant before CUDA graph capture.
    bool preflightPagedTwoPoolNonCausal(cudaStream_t stream, bool fp8Prefix);

    //! Read a retained prefix and independent FP16 action pages with one non-causal softmax.
    bool runPagedTwoPoolNonCausal(void const* qPtr, void const* prefixPool, int32_t const* prefixPages,
        void const* actionPool, int32_t const* actionPages, void* oPtr, int32_t const* cuQSeqLens,
        int32_t const* cuKVSeqLens, int32_t numPrefixFlatPages, int32_t maxPrefixPagesPerSeq,
        int32_t numActionFlatPages, int32_t maxActionPagesPerSeq, cudaStream_t stream, float attentionScale,
        bool fp8Prefix, float kScale, float vScale);

    //! Inputs for packed Q/O attention over the native NHD paged KV cache.
    struct PagedRaggedParams
    {
        void const* qPtr{};
        void const* pagedKVPoolPtr{};
        int32_t const* kvCachePageList{};
        void* oPtr{};
        int32_t const* cuQSeqLens{};
        int32_t const* cuKVSeqLens{};
        int32_t totalQSeqLen{};
        int32_t maxQSeqLen{};
        int32_t numFlatPages{};
        int32_t maxPagesPerSeq{};
        int32_t tokensPerPage{};
        cudaStream_t stream{};
        float attentionScale{};
        int32_t slidingWindowSize{INT_MAX};
        float const* attentionSinks{};
        int32_t const* bidirectionalBlockBegin{};
        int32_t const* bidirectionalBlockEnd{};
    };

    /**
     * @brief Runs causal, sliding-causal, or vision-block attention over packed Q/O and FP16 NHD paged KV.
     *
     * cuQSeqLens and cuKVSeqLens are monotonic prefix arrays of length batchSize + 1. cuQSeqLens starts at zero and
     * ends at totalQSeqLen. cuKVSeqLens contains the complete logical KV history, including prior cached tokens.
     * maxQSeqLen is the maximum adjacent difference in cuQSeqLens, not the packed buffer allocation capacity.
     * attentionSinks optionally points to device-resident FP32 sink logits indexed by query head; nullptr disables
     * sinks. bidirectionalBlockBegin and bidirectionalBlockEnd optionally contain inclusive, sequence-relative KV
     * intervals, indexed by packed query row [totalQSeqLen]. Text rows use the empty interval [-1, -2]. Vision blocks
     * must be contiguous and contain the query at KV position (kvLength - qLength + queryRow).
     */
    bool runPagedRagged(PagedRaggedParams const& inputs);

    //! Runs dense non-causal padded context attention with independent logical Q/KV lengths.
    //! D512 additionally needs [B, S_q] full-KV intervals from launchBuildPaddingBlockRanges().
    bool runPadding(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr, int32_t const* cuQSeqLens,
        int32_t const* cuKVSeqLens, cudaStream_t stream, float attentionScale,
        int32_t const* bidirectionalBlockBegin = nullptr, int32_t const* bidirectionalBlockEnd = nullptr);

    //! Runs Gemma4 vision-block attention: sliding-causal OR same-image-block.
    bool runVisionBlock(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr, int32_t const* cuKVSeqLens,
        int32_t const* bidirectionalBlockBegin, int32_t const* bidirectionalBlockEnd, cudaStream_t stream,
        float attentionScale, int32_t slidingWindowSize);

    //! Runs packed varlen, bidirectional ViT attention.
    bool run(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr, int32_t const* cuSeqLens,
        int32_t totalSeqLen, int32_t maxSeqLen, int32_t batchSize, cudaStream_t stream, float attentionScale);

private:
    int32_t mBatchSize{};
    int32_t mSeqLenQ{};
    int32_t mKVSeqLen{};
    int32_t mNumHeadsQ{};
    int32_t mNumHeadsKV{};
    int32_t mHeadDim{};
    bool mUseSmallD64{true};

#if defined(CUTE_DSL_FMHA_ENABLED)
    static detail::LazyKernelModule<fmha_v2_d64_Kernel_Module_t> sLLM_d64;
    static detail::LazyKernelModule<fmha_v2_d64_small_Kernel_Module_t> sLLM_d64Small;
    static detail::LazyKernelModule<fmha_v2_d128_Kernel_Module_t> sLLM_d128;
    static detail::LazyKernelModule<fmha_v2_d256_Kernel_Module_t> sLLM_d256;
    static detail::LazyKernelModule<fmha_v2_d512_Kernel_Module_t> sLLM_d512;
    static detail::LazyKernelModule<fmha_v2_d256_padding_Kernel_Module_t> sLLM_d256Padding;
    static detail::LazyKernelModule<fmha_v2_d64_sw_Kernel_Module_t> sLLM_d64Sw;
    static detail::LazyKernelModule<fmha_v2_d128_sw_Kernel_Module_t> sLLM_d128Sw;
    static detail::LazyKernelModule<fmha_v2_d256_sw_Kernel_Module_t> sLLM_d256Sw;
    static detail::LazyKernelModule<fmha_v2_d512_sw_Kernel_Module_t> sLLM_d512Sw;
    static detail::LazyKernelModule<fmha_v2_d256_bidirectional_Kernel_Module_t> sLLM_d256Bidirectional;
    static detail::LazyKernelModule<fmha_v2_d512_bidirectional_Kernel_Module_t> sLLM_d512Bidirectional;
    static detail::LazyKernelModule<fmha_v2_d64_paged_Kernel_Module_t> sLLM_d64Paged;
    static detail::LazyKernelModule<fmha_v2_d64_small_paged_Kernel_Module_t> sLLM_d64SmallPaged;
    static detail::LazyKernelModule<fmha_v2_d128_paged_Kernel_Module_t> sLLM_d128Paged;
    static detail::LazyKernelModule<fmha_v2_d256_paged_Kernel_Module_t> sLLM_d256Paged;
    static detail::LazyKernelModule<fmha_v2_d512_paged_Kernel_Module_t> sLLM_d512Paged;
    static detail::LazyKernelModule<fmha_v2_d64_sw_paged_Kernel_Module_t> sLLM_d64SwPaged;
    static detail::LazyKernelModule<fmha_v2_d128_sw_paged_Kernel_Module_t> sLLM_d128SwPaged;
    static detail::LazyKernelModule<fmha_v2_d256_sw_paged_Kernel_Module_t> sLLM_d256SwPaged;
    static detail::LazyKernelModule<fmha_v2_d512_sw_paged_Kernel_Module_t> sLLM_d512SwPaged;
#if CUDA_VERSION >= 12000
    static detail::LazyKernelModule<fmha_v2_d128_non_causal_paged_two_pool_Kernel_Module_t>
        sLLM_d128NonCausalPagedTwoPool;
    static detail::LazyKernelModule<fmha_v2_d128_non_causal_paged_two_pool_fp8_Kernel_Module_t>
        sLLM_d128NonCausalPagedTwoPoolFp8;
#endif // CUDA_VERSION >= 12000
    static detail::LazyKernelModule<fmha_v2_d256_vision_paged_ragged_Kernel_Module_t> sLLM_d256VisionPagedRagged;
    static detail::LazyKernelModule<fmha_v2_d512_vision_paged_ragged_Kernel_Module_t> sLLM_d512VisionPagedRagged;
    static detail::LazyKernelModule<fmha_v2_d64_paged_ragged_Kernel_Module_t> sLLM_d64PagedRagged;
    static detail::LazyKernelModule<fmha_v2_d128_paged_ragged_Kernel_Module_t> sLLM_d128PagedRagged;
    static detail::LazyKernelModule<fmha_v2_d256_paged_ragged_Kernel_Module_t> sLLM_d256PagedRagged;
    static detail::LazyKernelModule<fmha_v2_d512_paged_ragged_Kernel_Module_t> sLLM_d512PagedRagged;
    static detail::LazyKernelModule<fmha_v2_d64_sw_paged_ragged_Kernel_Module_t> sLLM_d64SwPagedRagged;
    static detail::LazyKernelModule<fmha_v2_d128_sw_paged_ragged_Kernel_Module_t> sLLM_d128SwPagedRagged;
    static detail::LazyKernelModule<fmha_v2_d256_sw_paged_ragged_Kernel_Module_t> sLLM_d256SwPagedRagged;
    static detail::LazyKernelModule<fmha_v2_d512_sw_paged_ragged_Kernel_Module_t> sLLM_d512SwPagedRagged;

    static detail::LazyKernelModule<fmha_v2_vit_d64_Kernel_Module_t> sViT_d64;
    static detail::LazyKernelModule<fmha_v2_vit_d72_Kernel_Module_t> sViT_d72;
    static detail::LazyKernelModule<fmha_v2_vit_d80_Kernel_Module_t> sViT_d80;
    static detail::LazyKernelModule<fmha_v2_vit_d96_Kernel_Module_t> sViT_d96;
    static detail::LazyKernelModule<fmha_v2_vit_d128_Kernel_Module_t> sViT_d128;
#endif // defined(CUTE_DSL_FMHA_ENABLED)
};

} // namespace trt_edgellm
