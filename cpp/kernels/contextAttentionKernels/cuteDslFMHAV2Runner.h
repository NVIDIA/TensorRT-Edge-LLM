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

#include <NvInferRuntime.h>
#include <climits>
#include <cstdint>

namespace trt_edgellm
{

//! Mask contracts implemented by the FMHA-v2 CuTe DSL context kernels.
enum class CuteDslFMHAV2MaskType
{
    kCAUSAL,
    kSLIDING_CAUSAL,
    kPADDING,
    kVISION_BLOCK
};

//! Runner for the CuTe DSL FMHA-v2 kernels.
//!
//! Dense LLM kernels consume separate BSND Q/K/V tensors, while native-paged LLM kernels consume
//! BSND Q/O and the Edge-LLM NHD paged KV pool. The special padding and vision-block kernels consume
//! dense, separate Q/K/V tensors, while ViT kernels consume packed, separate Q/K/V tensors. The
//! runner remains separate from CuteDslFMHARunner because the optimized SM100/101/110 family has a
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
        nvinfer1::DataType dataType, CuteDslFMHAV2MaskType maskType);

    //! Returns whether the target AOT family covers native FP16 Edge-LLM paged KV for this context shape.
    static bool canImplementPaged(int32_t numQHeads, int32_t numKVHeads, int32_t headSize, int32_t smVersion,
        nvinfer1::DataType dataType, CuteDslFMHAV2MaskType maskType);

    //! Returns whether the target AOT family covers this packed ViT shape.
    static bool canImplementViT(int32_t headSize, int32_t smVersion, nvinfer1::DataType dataType);

    //! Ensures the exact causal or sliding-causal LLM variant selected by run() is loaded.
    bool preflightLlm(cudaStream_t stream, int32_t slidingWindowSize = INT_MAX);

    //! Ensures the exact native-paged causal or sliding-causal LLM variant selected by runPaged() is loaded.
    bool preflightPaged(cudaStream_t stream, int32_t slidingWindowSize = INT_MAX);

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
    bool runPaged(void const* qPtr, void const* pagedKVPoolPtr, int32_t const* kvCachePageList, void* oPtr,
        int32_t const* cuQSeqLens, int32_t const* cuKVSeqLens, int32_t numFlatPages, int32_t maxPagesPerSeq,
        int32_t tokensPerPage, cudaStream_t stream, float attentionScale, int32_t slidingWindowSize = INT_MAX);

    //! Runs dense non-causal padded context attention with independent logical Q/KV lengths.
    bool runPadding(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr, int32_t const* cuQSeqLens,
        int32_t const* cuKVSeqLens, cudaStream_t stream, float attentionScale);

    //! Runs Gemma4 vision-block attention: sliding-causal OR same-image-block.
    bool runVisionBlock(void const* qPtr, void const* kPtr, void const* vPtr, void* oPtr, int32_t const* cuKVSeqLens,
        int32_t const* blockBegin, int32_t const* blockEnd, cudaStream_t stream, float attentionScale,
        int32_t slidingWindowSize);

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

    static detail::LazyKernelModule<fmha_v2_vit_d64_Kernel_Module_t> sViT_d64;
    static detail::LazyKernelModule<fmha_v2_vit_d72_Kernel_Module_t> sViT_d72;
    static detail::LazyKernelModule<fmha_v2_vit_d80_Kernel_Module_t> sViT_d80;
    static detail::LazyKernelModule<fmha_v2_vit_d128_Kernel_Module_t> sViT_d128;
#endif // defined(CUTE_DSL_FMHA_ENABLED)
};

} // namespace trt_edgellm
